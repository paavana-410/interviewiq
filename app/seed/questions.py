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
    # ── Tricky & Intelligent HR / Behavioral Questions ────────────────────────
    {
        "question_text": "Your manager asks you to bypass unit tests and security reviews to push a critical client demo before a board meeting tomorrow. How do you respond and handle this situation?",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "ethics & trade-offs",
        "difficulty": "hard",
        "rubrics": [
            {
                "criterion_name": "professional_integrity",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.35,
                "expected_concepts": ["security risk", "production integrity", "ethical stance", "risk communication"],
                "star_component": False,
                "scoring_guide": "1.0: Refuses blind compromise on critical security while offering pragmatic solutions.",
            },
            {
                "criterion_name": "pragmatic_mitigation",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.35,
                "expected_concepts": ["staging environment", "scoped feature flag", "smoke testing", "explicit sign-off"],
                "star_component": False,
                "scoring_guide": "1.0: Proposes isolated sandbox or partial demo without jeopardizing prod.",
            },
            {
                "criterion_name": "diplomatic_communication",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.3,
                "expected_concepts": ["empathy for business deadline", "respectful pushback", "clear escalation"],
                "star_component": True,
                "scoring_guide": "1.0: Calm, solution-oriented executive communication.",
            },
        ],
    },
    {
        "question_text": "During a high-stakes code review, a senior engineer harshly criticizes your pull request in front of the team, calling your approach fundamentally amateur. How do you respond and handle this constructively?",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "conflict & resilience",
        "difficulty": "hard",
        "rubrics": [
            {
                "criterion_name": "emotional_regulation",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.3,
                "expected_concepts": ["staying calm", "separating ego from code", "professional composure"],
                "star_component": False,
                "scoring_guide": "1.0: Acknowledges initial sting but focuses immediately on objective merits.",
            },
            {
                "criterion_name": "objective_inquiry",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.4,
                "expected_concepts": ["1-on-1 offline discussion", "seeking specific architectural feedback", "curiosity"],
                "star_component": False,
                "scoring_guide": "1.0: Takes conversation offline to understand technical rationale without defensiveness.",
            },
            {
                "criterion_name": "growth_outcome",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.3,
                "expected_concepts": ["improved implementation", "team psychological safety", "long-term relationship"],
                "star_component": True,
                "scoring_guide": "1.0: Concrete improvement in code and mutual respect established.",
            },
        ],
    },
    {
        "question_text": "Tell me about a catastrophic or costly mistake you made in a technical deliverable. What went wrong, when did you realize it, and how did you own up to it?",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "accountability",
        "difficulty": "hard",
        "rubrics": [
            {
                "criterion_name": "genuine_accountability",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.4,
                "expected_concepts": ["unfiltered ownership", "no blaming others", "prompt disclosure"],
                "star_component": True,
                "scoring_guide": "1.0: Real substantive mistake with zero deflection.",
            },
            {
                "criterion_name": "triage_and_rectification",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.35,
                "expected_concepts": ["rollback", "incident response", "stakeholder alert", "immediate mitigation"],
                "star_component": True,
                "scoring_guide": "1.0: Decisive emergency steps taken to minimize damage.",
            },
            {
                "criterion_name": "systemic_prevention",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.25,
                "expected_concepts": ["post-mortem", "automated guardrails", "ci/cd check", "process improvement"],
                "star_component": False,
                "scoring_guide": "1.0: Built guardrails so the entire team never repeats the error.",
            },
        ],
    },
    {
        "question_text": "In a team project, a peer is completely unresponsive and failing to deliver their assigned core module 48 hours before the deadline. How do you handle this situation?",
        "category": QuestionCategory.BEHAVIORAL,
        "domain": "team dynamics",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "empathy_and_checkin",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.3,
                "expected_concepts": ["check well-being", "direct private reachout", "assume positive intent"],
                "star_component": False,
                "scoring_guide": "1.0: Reaches out privately first to check for personal emergency or blocker.",
            },
            {
                "criterion_name": "triage_redistribution",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.4,
                "expected_concepts": ["scope reduction", "work redistribution", "pair programming", "contingency MVP"],
                "star_component": False,
                "scoring_guide": "1.0: Re-scopes requirements to salvage working MVP without panic.",
            },
            {
                "criterion_name": "transparent_escalation",
                "criterion_type": CriterionType.STAR_STRUCTURE,
                "weight": 0.3,
                "expected_concepts": ["manager heads-up", "factual non-accusatory update", "team accountability"],
                "star_component": True,
                "scoring_guide": "1.0: Involves instructor/manager factually if blocker persists.",
            },
        ],
    },
    {
        "question_text": "We have candidates from top-ranked universities with 9+ GPAs and big-tech internships. Why should our engineering team bet on you over them?",
        "category": QuestionCategory.HR,
        "domain": "self-awareness & pitch",
        "difficulty": "medium",
        "rubrics": [
            {
                "criterion_name": "differentiated_value",
                "criterion_type": CriterionType.RELEVANCE,
                "weight": 0.4,
                "expected_concepts": ["hands-on building", "scrappiness", "problem solving", "genuine curiosity"],
                "star_component": False,
                "scoring_guide": "1.0: Articulates unique strengths without putting down competitors.",
            },
            {
                "criterion_name": "grit_and_coachability",
                "criterion_type": CriterionType.CONCEPT_COVERAGE,
                "weight": 0.3,
                "expected_concepts": ["fast learner", "handling feedback", "perseverance", "hunger to contribute"],
                "star_component": False,
                "scoring_guide": "1.0: Demonstrates relentless work ethic and rapid adaptability.",
            },
            {
                "criterion_name": "company_alignment",
                "criterion_type": CriterionType.COMPLETENESS,
                "weight": 0.3,
                "expected_concepts": ["knowledge of company tech/mission", "immediate team contribution"],
                "star_component": False,
                "scoring_guide": "1.0: Ties strengths directly to engineering team needs.",
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
            qdata_copy = dict(qdata)           # don't mutate the module-level list
            rubrics_data = qdata_copy.pop("rubrics")
            q = Question(**qdata_copy)
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
