"""Tests for Coding & SQL IDE execution."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from app.services.code_executor import CHALLENGES, get_challenge_metadata, execute_challenge
from app.api.routes.interview import CodeRunRequest, CodeSubmitRequest, run_code, list_code_challenges, get_code_challenge
from app.database import get_session_local
from app.models.core import InterviewSession, SessionMode, SessionStatus


def test_list_code_challenges():
    data = list_code_challenges()
    assert len(data) >= 4
    ids = [d["id"] for d in data]
    assert "two_sum" in ids
    assert "valid_parentheses" in ids
    assert "sql_department_top_salary" in ids
    assert "sql_customer_orders_total" in ids
    for chal in data:
        assert chal["open_count"] == 3
        assert chal["hidden_count"] == 3
        assert len(chal["open_test_cases"]) == 3


def test_get_code_challenge_details_no_hidden_leaks():
    """Verify that get_challenge_metadata NEVER exposes hidden test case inputs or answers."""
    data = get_code_challenge("two_sum")
    assert data["title"] == "Two Sum (Target Pair Indices)"
    assert len(data["open_test_cases"]) == 3
    assert data["open_test_cases"][0]["input_display"] == "nums = [2, 7, 11, 15], target = 9"
    # Ensure expected_data is not leaked
    for tc in data["open_test_cases"]:
        assert "expected_data" not in tc
        assert "input_data" not in tc


def test_run_python_code_two_sum():
    code = CHALLENGES["two_sum"]["starter_code"]["python"]
    req = CodeRunRequest(challenge_id="two_sum", language="python", source_code=code)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3
    assert len(data["results"]) == 3


def test_run_javascript_code_two_sum():
    code = CHALLENGES["two_sum"]["starter_code"]["javascript"]
    req = CodeRunRequest(challenge_id="two_sum", language="javascript", source_code=code)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3


def test_run_python_valid_parentheses():
    code = CHALLENGES["valid_parentheses"]["starter_code"]["python"]
    req = CodeRunRequest(challenge_id="valid_parentheses", language="python", source_code=code)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3


def test_run_javascript_valid_parentheses():
    code = CHALLENGES["valid_parentheses"]["starter_code"]["javascript"]
    req = CodeRunRequest(challenge_id="valid_parentheses", language="javascript", source_code=code)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3


def test_run_java_code_endpoint_handling():
    """Verify Java starter code exists and execution engine handles Java requests safely."""
    java_code_two_sum = CHALLENGES["two_sum"]["starter_code"]["java"]
    assert "class Solution" in java_code_two_sum
    assert "twoSum" in java_code_two_sum

    req = CodeRunRequest(challenge_id="two_sum", language="java", source_code=java_code_two_sum)
    data = run_code(req)
    assert data["challenge_id"] == "two_sum"
    assert data["language"] == "java"
    assert data["total_tests"] == 3
    # If JDK is not installed on the system, it cleanly reports JDK Not Found rather than crashing
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert "test_case_id" in r
        assert "status" in r


def test_run_sql_department_top_salary():
    query = CHALLENGES["sql_department_top_salary"]["starter_code"]["sql"]
    req = CodeRunRequest(challenge_id="sql_department_top_salary", language="sql", source_code=query)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3


def test_run_sql_customer_orders_total():
    query = CHALLENGES["sql_customer_orders_total"]["starter_code"]["sql"]
    req = CodeRunRequest(challenge_id="sql_customer_orders_total", language="sql", source_code=query)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3


def test_submit_code_all_six_cases():
    # Submit evaluation tests all 6 cases (3 open + 3 hidden)
    code = CHALLENGES["two_sum"]["starter_code"]["python"]
    data = execute_challenge("two_sum", "python", code, mode="submit")
    assert data["score_percent"] == 100
    assert data["all_passed"] is True
    assert data["passed_tests"] == 6
    assert data["total_tests"] == 6

    # Test wrong code returns correct partial/zero score
    wrong_code = "def two_sum(nums, target): return [0, 0]"
    wrong_data = execute_challenge("two_sum", "python", wrong_code, mode="submit")
    assert wrong_data["all_passed"] is False
    assert wrong_data["passed_tests"] < 6


def test_submit_sql_customer_orders_all_six_cases():
    query = CHALLENGES["sql_customer_orders_total"]["starter_code"]["sql"]
    data = execute_challenge("sql_customer_orders_total", "sql", query, mode="submit")
    assert data["score_percent"] == 100
    assert data["all_passed"] is True
    assert data["passed_tests"] == 6
    assert data["total_tests"] == 6


def test_rate_limiting_runs_per_session():
    from app.database import Base
    db = get_session_local()()
    try:
        Base.metadata.create_all(bind=db.get_bind())
        session = InterviewSession(mode=SessionMode.PRACTICE, status=SessionStatus.ACTIVE)
        session.meta_json = {"code_run_count": 15}
        db.add(session)
        db.commit()
        db.refresh(session)

        req = CodeRunRequest(
            challenge_id="two_sum",
            language="python",
            source_code="def two_sum(n, t): return []",
            session_id=session.session_id,
        )
        with pytest.raises(HTTPException) as exc_info:
            run_code(req, db=db)
        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.detail
    finally:
        db.close()


def test_syntax_and_compilation_error_classification():
    # 1. Python Syntax Error
    bad_python = "def two_sum(nums, target):\n    return [0 1]" # syntax error
    req1 = CodeRunRequest(challenge_id="two_sum", language="python", source_code=bad_python)
    res1 = run_code(req1)
    assert res1["all_passed"] is False
    assert res1["results"][0]["status"] == "Compilation Error"
    assert "SyntaxError" in res1["results"][0]["error"]

    # 2. Python Runtime Error (e.g. ZeroDivisionError)
    runtime_python = "def two_sum(nums, target):\n    return 10 / 0"
    req2 = CodeRunRequest(challenge_id="two_sum", language="python", source_code=runtime_python)
    res2 = run_code(req2)
    assert res2["all_passed"] is False
    assert res2["results"][0]["status"] == "Runtime Error"
    assert "division by zero" in res2["results"][0]["error"].lower()

    # 3. SQL Syntax Error
    bad_sql = "SELEC * FORM invalid table syntax;;;"
    req3 = CodeRunRequest(challenge_id="sql_department_top_salary", language="sql", source_code=bad_sql)
    res3 = run_code(req3)
    assert res3["all_passed"] is False
    assert res3["results"][0]["status"] == "SQL Syntax Error"
    assert "syntax error" in res3["results"][0]["error"].lower()
