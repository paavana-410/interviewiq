"""Agreement metrics: Cohen's Kappa, Fleiss' Kappa, Weighted F1, Krippendorff's Alpha.

Usage:
    from app.analysis.agreement import compute_agreement
    report = compute_agreement(db, response_ids=[1, 2, 3])
"""
from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.models.core import ContentEvaluation, HumanAnnotation


def _cohen_kappa(a: list[float], b: list[float], bins: int = 5) -> float:
    """Cohen's Kappa for two raters on continuous scores (discretized into bins)."""
    from sklearn.metrics import cohen_kappa_score  # type: ignore

    # Discretize to bins
    edges = np.linspace(0, 1, bins + 1)
    a_disc = np.digitize(a, edges[1:-1])
    b_disc = np.digitize(b, edges[1:-1])
    return float(cohen_kappa_score(a_disc, b_disc))


def _fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Fleiss' Kappa for N items × K categories matrix."""
    n, k = ratings_matrix.shape
    N = ratings_matrix.sum(axis=1)
    p_j = ratings_matrix.sum(axis=0) / (N.sum())
    P_e = (p_j ** 2).sum()
    P_i = ((ratings_matrix ** 2).sum(axis=1) - N) / (N * (N - 1))
    P_bar = P_i.mean()
    if P_e == 1:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def _bootstrap_ci(
    data: list[float],
    stat_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a scalar statistic."""
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    boot_stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (
        float(np.quantile(boot_stats, alpha)),
        float(np.quantile(boot_stats, 1 - alpha)),
    )


def compute_kappa_human_vs_llm(
    db: Session,
    criterion: str,
    bins: int = 5,
) -> dict:
    """Compare human annotation scores vs LLM content scores for a criterion."""
    annotations = (
        db.query(HumanAnnotation).filter(HumanAnnotation.criterion == criterion).all()
    )
    if not annotations:
        return {"error": "No human annotations found", "criterion": criterion}

    human_scores: list[float] = []
    llm_scores: list[float] = []

    for ann in annotations:
        ce = (
            db.query(ContentEvaluation)
            .filter(ContentEvaluation.response_id == ann.response_id)
            .order_by(ContentEvaluation.created_at.desc())
            .first()
        )
        if ce is None:
            continue
        # Pull the matching criterion score from evidence_json
        llm_score = _extract_criterion_score(ce, criterion)
        if llm_score is None:
            continue
        human_scores.append(ann.score)
        llm_scores.append(llm_score)

    if len(human_scores) < 2:
        return {"error": "Not enough paired scores", "criterion": criterion}

    kappa = _cohen_kappa(human_scores, llm_scores, bins=bins)
    ci_lo, ci_hi = _bootstrap_ci(
        [abs(h - l) for h, l in zip(human_scores, llm_scores)],
        stat_fn=np.mean,
    )

    return {
        "criterion": criterion,
        "n": len(human_scores),
        "cohen_kappa": kappa,
        "mean_absolute_error": float(np.mean(np.abs(np.array(human_scores) - np.array(llm_scores)))),
        "bootstrap_mae_ci_95": [ci_lo, ci_hi],
    }


def _extract_criterion_score(ce: ContentEvaluation, criterion: str) -> float | None:
    """Pull a per-criterion score out of evidence_json."""
    if not ce.evidence_json:
        return None
    for c in ce.evidence_json.get("criteria", []):
        if criterion.lower() in c.get("criterion_name", "").lower():
            return c.get("score")
    return None


def weighted_f1_per_criterion(
    db: Session,
    criterion: str,
    threshold: float = 0.7,
) -> dict:
    """Weighted F1: treat score >= threshold as positive class."""
    from sklearn.metrics import f1_score  # type: ignore

    annotations = (
        db.query(HumanAnnotation).filter(HumanAnnotation.criterion == criterion).all()
    )
    if not annotations:
        return {"error": "No annotations", "criterion": criterion}

    y_true, y_pred = [], []
    for ann in annotations:
        ce = (
            db.query(ContentEvaluation)
            .filter(ContentEvaluation.response_id == ann.response_id)
            .order_by(ContentEvaluation.created_at.desc())
            .first()
        )
        if ce is None:
            continue
        llm_score = _extract_criterion_score(ce, criterion)
        if llm_score is None:
            continue
        y_true.append(int(ann.score >= threshold))
        y_pred.append(int(llm_score >= threshold))

    if len(y_true) < 2:
        return {"error": "Not enough data", "criterion": criterion}

    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "criterion": criterion,
        "n": len(y_true),
        "threshold": threshold,
        "weighted_f1": float(f1),
    }
