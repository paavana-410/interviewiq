"""Tests for Coding & SQL IDE execution."""
from __future__ import annotations

from app.services.code_executor import CHALLENGES, get_challenge_metadata, execute_challenge
from app.api.routes.interview import CodeRunRequest, CodeSubmitRequest, run_code, list_code_challenges, get_code_challenge


def test_list_code_challenges():
    data = list_code_challenges()
    assert len(data) >= 2
    ids = [d["id"] for d in data]
    assert "two_sum" in ids
    assert "sql_department_top_salary" in ids
    for chal in data:
        assert chal["open_count"] == 3
        assert chal["hidden_count"] == 3
        assert len(chal["open_test_cases"]) == 3


def test_get_code_challenge_details():
    data = get_code_challenge("two_sum")
    assert data["title"] == "Two Sum (Target Pair Indices)"
    assert len(data["open_test_cases"]) == 3
    assert data["open_test_cases"][0]["input_display"] == "nums = [2, 7, 11, 15], target = 9"


def test_run_python_code_endpoint():
    code = CHALLENGES["two_sum"]["starter_code"]["python"]
    req = CodeRunRequest(challenge_id="two_sum", language="python", source_code=code)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3
    assert len(data["results"]) == 3


def test_run_sql_query_endpoint():
    query = CHALLENGES["sql_department_top_salary"]["starter_code"]["sql"]
    req = CodeRunRequest(challenge_id="sql_department_top_salary", language="sql", source_code=query)
    data = run_code(req)
    assert data["all_passed"] is True
    assert data["passed_tests"] == 3
    assert data["total_tests"] == 3


def test_submit_code_evaluation():
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
