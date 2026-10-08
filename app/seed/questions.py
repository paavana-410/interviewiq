"""Seed data — 10 questions (HR, behavioral, technical) with rubrics.

Run via:  python -m app.seed.questions
"""
from __future__ import annotations

from app.database import get_session_local
from app.models.core import CriterionType, Question, QuestionCategory, Rubric

SEED_QUESTIONS = [
    # ── HR ────────────────────────────────────────────────────────────────────
    {
        "question_text": "Tell me about yourself.",
        "category": QuestionCategory.HR,
        "domain": "general",
        "difficulty": "easy",
        "rubrics": [
            {
                "criterion_name": "relevance",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.4,
                "expected_concepts": ["education", "skills", "experience", "career goals"],
                "star_component": False,
                "scoring_guide": (
                    "1.0: Covers education, skills, relevant experience, and goal alignment. "
                    "0.5: Covers 2–3 elements. 0.0: Generic or irrelevant."
                ),
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.3,
                "expected_concepts": ["past", "present", "future"],
                "star_component": False,
                "scoring_guide": (
                    "1.0: Structured past→present→future narrative. "
                    "0.5: Missing one dimension. 0.0: No structure."
                ),
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["technical skills", "soft skills", "achievements"],
                "star_component": False,
                "scoring_guide": "1.0: ≥3 concepts. 0.5: 1–2 concepts. 0.0: none.",
            },
        ],
    },
    {
        "question_text": "What is your greatest weakness?",
        "category": QuestionCategory.HR,
        "domain": "general",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "self_awareness",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.4,
                "expected_concepts": ["genuine weakness", "honest self-reflection"],
                "star_component": False,
                "scoring_guide": "1.0: Real weakness named clearly. 0.0: Cliché non-answer.",
            },
            {
                "criterion_name": "growth_plan",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.6,
                "expected_concepts": ["mitigation strategy", "improvement steps", "progress made"],
                "star_component": False,
                "scoring_guide": "1.0: Clear mitigation plan with evidence of progress.",
            },
        ],
    },
    # ── Behavioral ────────────────────────────────────────────────────────────
    {
        "question_text": "Tell me about a time you handled a conflict with a teammate.",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "teamwork",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "star_situation",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.15,
                "expected_concepts": ["context", "team", "conflict scenario"],
                "star_component": True,
                "scoring_guide": "1.0: Clear situation described. 0.0: Missing.",
            },
            {
                "criterion_name": "star_task",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.15,
                "expected_concepts": ["role", "responsibility"],
                "star_component": True,
                "scoring_guide": "1.0: Candidate's specific task stated. 0.0: Missing.",
            },
            {
                "criterion_name": "star_action",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.4,
                "expected_concepts": ["communication", "empathy", "steps taken"],
                "star_component": True,
                "scoring_guide": "1.0: Specific, personal actions described. 0.0: Vague.",
            },
            {
                "criterion_name": "star_result",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.3,
                "expected_concepts": ["outcome", "resolution", "learning"],
                "star_component": True,
                "scoring_guide": "1.0: Concrete positive outcome stated. 0.0: Missing.",
            },
        ],
    },
    {
        "question_text": "Describe a situation where you had to meet a tight deadline.",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "time management",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "star_situation",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.15,
                "expected_concepts": ["project", "deadline", "pressure"],
                "star_component": True,
                "scoring_guide": "1.0: Specific context given.",
            },
            {
                "criterion_name": "star_action",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.5,
                "expected_concepts": ["prioritization", "planning", "execution"],
                "star_component": True,
                "scoring_guide": "1.0: Concrete steps. 0.5: Vague strategy. 0.0: None.",
            },
            {
                "criterion_name": "star_result",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.35,
                "expected_concepts": ["deadline met", "outcome", "lesson"],
                "star_component": True,
                "scoring_guide": "1.0: Clear result. 0.0: No mention.",
            },
        ],
    },
    {
        "question_text": "Give an example of when you showed leadership.",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "leadership",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "star_action",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.5,
                "expected_concepts": ["delegating", "motivating", "decision-making"],
                "star_component": True,
                "scoring_guide": "1.0: Explicit leadership behaviours described.",
            },
            {
                "criterion_name": "star_result",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.3,
                "expected_concepts": ["team success", "measurable outcome"],
                "star_component": True,
                "scoring_guide": "1.0: Measurable positive result.",
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.2,
                "expected_concepts": ["initiative", "accountability"],
                "star_component": False,
                "scoring_guide": "Score by concept coverage.",
            },
        ],
    },
    # ── Technical ─────────────────────────────────────────────────────────────
    {
        "question_text": "Explain the difference between a stack and a queue.",
        "category": QuestionCategory.TECHNICAL,
        "domain": "data structures",
        "difficulty": "easy",
        "rubrics": [
            {
                "criterion_name": "technical_correctness",
                "criterion_type": CriterionType.CORRECTNESS,
                "weight": 0.5,
                "expected_concepts": ["LIFO", "FIFO", "push", "pop", "enqueue", "dequeue"],
                "star_component": False,
                "scoring_guide": (
                    "1.0: LIFO vs FIFO defined, operations named. "
                    "0.5: One defined correctly. 0.0: Incorrect."
                ),
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["use cases", "implementation"],
                "star_component": False,
                "scoring_guide": "1.0: Use cases given for both. 0.5: One only.",
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.2,
                "expected_concepts": ["time complexity", "example"],
                "star_component": False,
                "scoring_guide": "1.0: Complexity and example included.",
            },
        ],
    },
    {
        "question_text": "What is a RESTful API and what are its key constraints?",
        "category": QuestionCategory.TECHNICAL,
        "domain": "web development",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "technical_correctness",
                "criterion_type": CriterionType.CORRECTNESS,
                "weight": 0.5,
                "expected_concepts": [
                    "stateless", "client-server", "uniform interface",
                    "cacheable", "layered system", "code on demand",
                ],
                "star_component": False,
                "scoring_guide": "1.0: ≥4 constraints named correctly.",
            },
            {
                "criterion_name": "relevance",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.3,
                "expected_concepts": ["HTTP methods", "resources", "JSON"],
                "star_component": False,
                "scoring_guide": "1.0: Practical REST usage explained.",
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.2,
                "expected_concepts": ["example endpoint"],
                "star_component": False,
                "scoring_guide": "1.0: Example given.",
            },
        ],
    },
    {
        "question_text": "Explain how a hash map works and its time complexity.",
        "category": QuestionCategory.TECHNICAL,
        "domain": "data structures",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "technical_correctness",
                "criterion_type": CriterionType.CORRECTNESS,
                "weight": 0.5,
                "expected_concepts": [
                    "hash function", "buckets", "collision", "O(1) average",
                    "chaining", "open addressing",
                ],
                "star_component": False,
                "scoring_guide": "1.0: Hash function + collision handling + complexity correct.",
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["worst case", "load factor", "rehashing"],
                "star_component": False,
                "scoring_guide": "1.0: ≥2 advanced concepts covered.",
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.2,
                "expected_concepts": ["use case"],
                "star_component": False,
                "scoring_guide": "1.0: Real-world use case mentioned.",
            },
        ],
    },
    {
        "question_text": "What is the difference between process and thread?",
        "category": QuestionCategory.TECHNICAL,
        "domain": "operating systems",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "technical_correctness",
                "criterion_type": CriterionType.CORRECTNESS,
                "weight": 0.5,
                "expected_concepts": [
                    "independent memory space", "shared memory",
                    "context switching", "IPC", "lightweight",
                ],
                "star_component": False,
                "scoring_guide": "1.0: Key differences correctly stated.",
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["concurrency", "parallelism", "synchronization"],
                "star_component": False,
                "scoring_guide": "1.0: ≥2 concurrency concepts mentioned.",
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.2,
                "expected_concepts": ["example language or OS"],
                "star_component": False,
                "scoring_guide": "1.0: Concrete example given.",
            },
        ],
    },
    {
        "question_text": "How does garbage collection work in Java or Python?",
        "category": QuestionCategory.TECHNICAL,
        "domain": "programming languages",
        "difficulty": "hard",
        "rubrics": [
            {
                "criterion_name": "technical_correctness",
                "criterion_type": CriterionType.CORRECTNESS,
                "weight": 0.5,
                "expected_concepts": [
                    "reference counting", "mark-and-sweep", "generational GC",
                    "heap", "GC roots",
                ],
                "star_component": False,
                "scoring_guide": "1.0: At least one mechanism fully correct.",
            },
            {
                "criterion_name": "concept_coverage",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["memory leak", "GC pause", "cyclic references"],
                "star_component": False,
                "scoring_guide": "1.0: ≥2 advanced concepts covered.",
            },
            {
                "criterion_name": "completeness",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.2,
                "expected_concepts": ["language-specific detail"],
                "star_component": False,
                "scoring_guide": "1.0: Java OR Python specifics mentioned.",
            },
        ],
    },
]


def seed(db=None) -> None:
    """Insert seed questions + rubrics if not already present."""
    from app.database import get_session_local
    close = False
    if db is None:
        db = get_session_local()()
        close = True

    try:
        existing = db.query(Question).count()
        if existing >= len(SEED_QUESTIONS):
            print(f"Seed skipped: {existing} questions already in DB.")
            return

        for qdata in SEED_QUESTIONS:
            rubrics_data = qdata.pop("rubrics")
            q = Question(**qdata)
            db.add(q)
            db.flush()

            for rdata in rubrics_data:
                r = Rubric(question_id=q.question_id, **rdata)
                db.add(r)

        db.commit()
        print(f"Seeded {len(SEED_QUESTIONS)} questions with rubrics.")
    finally:
        if close:
            db.close()


if __name__ == "__main__":
    seed()
