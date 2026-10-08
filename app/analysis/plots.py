"""Analysis plots for audit benchmark results.

All plots are saved to the experiment directory.
Requires: matplotlib, pandas, numpy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_score_deltas(
    df: pd.DataFrame,
    out_dir: Path,
    perturbation_col: str = "variant_type",
    delta_col: str = "delta",
    title: str = "Content Score Δ per Perturbation Type",
) -> Path:
    """Box plot of score deltas grouped by perturbation type."""
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df.groupby(perturbation_col)[delta_col].apply(list)
    ax.boxplot(groups.values, labels=groups.index, patch_artist=True)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Content Score Delta")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out = out_dir / "score_deltas_per_perturbation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_repeatability(
    scores_per_run: list[float],
    out_dir: Path,
    title: str = "Repeated-Run Score Distribution",
) -> Path:
    """Violin/scatter plot of scores across N repeated runs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(1, len(scores_per_run) + 1))
    ax.plot(x, scores_per_run, "o-", markersize=8, color="steelblue")
    ax.axhline(y=np.mean(scores_per_run), color="orange", linestyle="--", label=f"mean={np.mean(scores_per_run):.3f}")
    ax.fill_between(
        x,
        np.mean(scores_per_run) - np.std(scores_per_run),
        np.mean(scores_per_run) + np.std(scores_per_run),
        alpha=0.2,
        color="orange",
        label=f"±1σ ({np.std(scores_per_run):.3f})",
    )
    ax.set_title(title)
    ax.set_xlabel("Run #")
    ax.set_ylabel("Overall Score")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "repeatability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_model_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    model_col: str = "model",
    metric_col: str = "overall_score",
    title: str = "Score Distribution by LLM Model",
) -> Path:
    """Bar chart + error bars comparing multiple judge models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = df.groupby(model_col)[metric_col].agg(["mean", "std"]).reset_index()
    ax.bar(summary[model_col], summary["mean"], yerr=summary["std"], capsize=5, color="cornflowerblue")
    ax.set_title(title)
    ax.set_xlabel("LLM Model")
    ax.set_ylabel("Mean Overall Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = out_dir / "model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_audit_summary(
    audit_metrics: dict,
    thresholds: dict,
    out_dir: Path,
    title: str = "Audit Metric Summary",
) -> Path:
    """Horizontal bar chart of all 5 audit metrics vs their thresholds."""
    names = list(audit_metrics.keys())
    values = [audit_metrics[n] for n in names]
    thresh = [thresholds.get(n, 0.1) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=["green" if v <= t else "red" for v, t in zip(values, thresh)])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    # Plot threshold markers
    for i, t in enumerate(thresh):
        ax.axvline(x=t, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(t + 0.002, i, f"thresh={t}", va="center", fontsize=8, color="black")
    ax.set_title(title)
    ax.set_xlabel("Metric Value")
    plt.tight_layout()
    out = out_dir / "audit_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
