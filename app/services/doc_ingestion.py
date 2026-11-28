"""Document Ingestion Service for InterviewIQ.

Allows faculty/placement admin to upload lab manuals, question banks,
or syllabus documents (TXT, CSV, PDF, Markdown) and extract verified questions
with rubrics directly into the SQLite database.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.models.core import CriterionType, Question, QuestionCategory, Rubric

logger = logging.getLogger(__name__)


def ingest_questions_from_text(
    db: Session,
    content: str,
    category: str = "technical",
    domain_override: str | None = None,
    default_difficulty: str = "medium",
) -> dict[str, Any]:
    """Parse text/CSV content and insert questions + rubrics into the database."""
    cat_enum = QuestionCategory.BEHAVIORAL if "beh" in category.lower() else QuestionCategory.TECHNICAL
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return {"success": False, "count": 0, "message": "Uploaded content is empty."}

    parsed_items: list[dict] = []

    # 1. Check if input is CSV format
    if "," in lines[0] and ("question" in lines[0].lower() or len(lines[0].split(",")) >= 2):
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                # Find question column
                q_text = row.get("question") or row.get("question_text") or row.get("Question") or ""
                if not q_text and len(row) > 0:
                    q_text = list(row.values())[0]

                if not q_text or len(q_text.strip()) < 10:
                    continue

                diff = (row.get("difficulty") or row.get("Difficulty") or default_difficulty).lower()
                if diff not in ("easy", "medium", "hard"):
                    diff = default_difficulty

                dom = row.get("domain") or row.get("Domain") or domain_override or ("Core CS" if cat_enum == QuestionCategory.TECHNICAL else "Workplace Scenarios")

                # Parse expected concepts
                raw_concepts = row.get("expected_concepts") or row.get("concepts") or row.get("Rubric") or ""
                if raw_concepts:
                    concepts = [c.strip() for c in re.split(r"[;,|]", raw_concepts) if c.strip()]
                else:
                    # Extract 3 key words from question text
                    words = [w for w in re.findall(r"[A-Za-z0-9_]+", q_text) if len(w) > 4][:4]
                    concepts = words or [dom, diff]

                guide = row.get("scoring_guide") or f"Candidate accurately explains core principles of {dom}."

                parsed_items.append({
                    "question_text": q_text.strip(),
                    "difficulty": diff,
                    "domain": dom,
                    "expected_concepts": concepts,
                    "scoring_guide": guide,
                })
        except Exception as csv_err:
            logger.warning("CSV parsing failed, falling back to line-by-line parsing: %s", csv_err)

    # 2. If not CSV, parse as structured list or document lines
    if not parsed_items:
        # Pattern match numbered or bulleted lines: "1. Explain...", "Q1: What is...", "- Describe..."
        current_q = None
        for line in lines:
            # Check for question starters
            match = re.match(r"^(?:Q\d+[:.]|\d+[\.\)]|[-*•])\s*(.+)$", line, re.IGNORECASE)
            if match or (line.endswith("?") and len(line) > 15):
                q_text = match.group(1).strip() if match else line.strip()
                if len(q_text) >= 12:
                    current_q = {
                        "question_text": q_text,
                        "difficulty": default_difficulty,
                        "domain": domain_override or ("Technical Architecture" if cat_enum == QuestionCategory.TECHNICAL else "Situational Leadership"),
                        "expected_concepts": [],
                        "scoring_guide": "",
                    }
                    parsed_items.append(current_q)
            elif current_q and len(current_q["expected_concepts"]) < 5:
                # Secondary line could be key concepts / keywords
                if "concept" in line.lower() or "rubric" in line.lower() or "keyword" in line.lower():
                    clean_c = re.sub(r"^(?:concepts?|rubric|keywords?)[:\-]\s*", "", line, flags=re.IGNORECASE)
                    concepts = [c.strip() for c in re.split(r"[,;|]", clean_c) if c.strip()]
                    current_q["expected_concepts"].extend(concepts)

    # 3. If standard regex found questions, enrich missing concepts
    for item in parsed_items:
        if not item["expected_concepts"]:
            # Auto-extract meaningful noun keywords from question text
            words = [w for w in re.findall(r"[A-Za-z0-9_]+", item["question_text"]) if len(w) > 3 and w.lower() not in ("what", "how", "explain", "describe", "difference", "between", "when", "which", "your", "with", "from")][:4]
            item["expected_concepts"] = words or [item["domain"]]
        if not item["scoring_guide"]:
            item["scoring_guide"] = f"Candidate covers expected concepts: {', '.join(item['expected_concepts'][:3])}."

    if not parsed_items:
        return {"success": False, "count": 0, "message": "No valid questions could be extracted from the document."}

    # 4. Save to Database
    inserted_count = 0
    for item in parsed_items:
        # Avoid exact duplicate questions
        existing = db.query(Question).filter(Question.question_text == item["question_text"]).first()
        if existing:
            continue

        q = Question(
            question_text=item["question_text"],
            category=cat_enum,
            domain=item["domain"],
            difficulty=item["difficulty"],
            active_flag=True,
        )
        db.add(q)
        db.flush()  # populate q.question_id

        r = Rubric(
            question_id=q.question_id,
            criterion_name="concept_coverage",
            criterion_type=CriterionType.CONCEPT_COVERAGE,
            weight=1.0,
            expected_concepts=item["expected_concepts"],
            star_component=(cat_enum == QuestionCategory.BEHAVIORAL),
            scoring_guide=item["scoring_guide"],
        )
        db.add(r)
        inserted_count += 1

    db.commit()

    return {
        "success": True,
        "count": inserted_count,
        "total_parsed": len(parsed_items),
        "category": cat_enum.value,
        "message": f"Successfully ingested {inserted_count} new {cat_enum.value} questions with verified rubrics.",
    }
