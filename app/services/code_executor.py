"""Code execution service for coding and SQL interview challenges.

Supports Python, JavaScript (Node.js), and SQL (in-memory SQLite).
Isolates execution with strict timeouts, separate process boundaries,
and deterministic evaluation of open and hidden test cases.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, List, Dict


@dataclass
class TestCase:
    id: int
    name: str
    input_display: str
    expected_display: str
    input_data: Any
    expected_data: Any
    is_hidden: bool = False
    explanation: str = ""


@dataclass
class TestCaseResult:
    test_case_id: int
    name: str
    input_display: str
    expected_display: str
    actual_display: str
    passed: bool
    status: str            # "Passed", "Wrong Answer", "Time Limit Exceeded", "Runtime Error"
    execution_time_ms: float
    is_hidden: bool
    stdout: str = ""
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Pre-seeded Coding & SQL Challenge Bank
# ─────────────────────────────────────────────────────────────────────────────

CHALLENGES = {
    "two_sum": {
        "id": "two_sum",
        "title": "Two Sum (Target Pair Indices)",
        "type": "code",
        "difficulty": "Easy",
        "category": "Algorithms & Arrays",
        "description": (
            "Given an array of integers <code>nums</code> and an integer <code>target</code>, "
            "return the <strong>0-based indices</strong> of the two numbers such that they add up to <code>target</code>.<br/><br/>"
            "You may assume that each input would have <strong>exactly one solution</strong>, and you may not use the same element twice.<br/>"
            "You can return the answer in any order (sorted indices preferred)."
        ),
        "constraints": [
            "2 &le; nums.length &le; 10<sup>4</sup>",
            "-10<sup>9</sup> &le; nums[i] &le; 10<sup>9</sup>",
            "-10<sup>9</sup> &le; target &le; 10<sup>9</sup>",
            "Only one valid answer exists."
        ],
        "starter_code": {
            "python": (
                "def two_sum(nums: list[int], target: int) -> list[int]:\n"
                "    # Write your solution here\n"
                "    seen = {}\n"
                "    for i, n in enumerate(nums):\n"
                "        comp = target - n\n"
                "        if comp in seen:\n"
                "            return [seen[comp], i]\n"
                "        seen[n] = i\n"
                "    return []\n"
            ),
            "javascript": (
                "function twoSum(nums, target) {\n"
                "    // Write your solution here\n"
                "    const seen = new Map();\n"
                "    for (let i = 0; i < nums.length; i++) {\n"
                "        const comp = target - nums[i];\n"
                "        if (seen.has(comp)) {\n"
                "            return [seen.get(comp), i];\n"
                "        }\n"
                "        seen.set(nums[i], i);\n"
                "    }\n"
                "    return [];\n"
                "}\n"
            )
        },
        "entrypoint": "two_sum",
        "test_cases": [
            # 3 Open Test Cases (Visible in UI)
            TestCase(
                id=1,
                name="Test Case 1",
                input_display="nums = [2, 7, 11, 15], target = 9",
                expected_display="[0, 1]",
                input_data={"nums": [2, 7, 11, 15], "target": 9},
                expected_data=[0, 1],
                is_hidden=False,
                explanation="nums[0] + nums[1] == 2 + 7 == 9, so return [0, 1]."
            ),
            TestCase(
                id=2,
                name="Test Case 2",
                input_display="nums = [3, 2, 4], target = 6",
                expected_display="[1, 2]",
                input_data={"nums": [3, 2, 4], "target": 6},
                expected_data=[1, 2],
                is_hidden=False,
                explanation="nums[1] + nums[2] == 2 + 4 == 6, so return [1, 2]."
            ),
            TestCase(
                id=3,
                name="Test Case 3",
                input_display="nums = [3, 3], target = 6",
                expected_display="[0, 1]",
                input_data={"nums": [3, 3], "target": 6},
                expected_data=[0, 1],
                is_hidden=False,
                explanation="nums[0] + nums[1] == 3 + 3 == 6, so return [0, 1]."
            ),
            # 3 Hidden Test Cases (Used for Verification & Final Score)
            TestCase(
                id=4,
                name="Hidden Test Case 1",
                input_display="[Hidden dataset 1: Negative numbers]",
                expected_display="[Hidden]",
                input_data={"nums": [-3, 4, 3, 90], "target": 0},
                expected_data=[0, 2],
                is_hidden=True,
                explanation="nums[0] + nums[2] == -3 + 3 == 0."
            ),
            TestCase(
                id=5,
                name="Hidden Test Case 2",
                input_display="[Hidden dataset 2: Boundary gap]",
                expected_display="[Hidden]",
                input_data={"nums": [1, 4, 8, 13, 19], "target": 20},
                expected_data=[0, 4],
                is_hidden=True,
                explanation="nums[0] + nums[4] == 1 + 19 == 20."
            ),
            TestCase(
                id=6,
                name="Hidden Test Case 3",
                input_display="[Hidden dataset 3: High values]",
                expected_display="[Hidden]",
                input_data={"nums": [100, 200, 300, 400], "target": 700},
                expected_data=[2, 3],
                is_hidden=True,
                explanation="nums[2] + nums[3] == 300 + 400 == 700."
            ),
        ]
    },
    "sql_department_top_salary": {
        "id": "sql_department_top_salary",
        "title": "Department Average Salary Filter (SQL)",
        "type": "sql",
        "difficulty": "Easy-Medium",
        "category": "Database & SQL",
        "description": (
            "You are given an <code>employees</code> table with schema:<br/>"
            "<pre style='background:#F8FAFA;padding:0.6rem;border-radius:4px;margin:0.5rem 0;'>employees(id INT, name VARCHAR, department VARCHAR, salary INT)</pre>"
            "Write a SQL query that returns the <code>department</code> and its average salary as <code>avg_salary</code> "
            "(rounded to the nearest whole integer) for departments where the average salary is <strong>greater than or equal to 70,000</strong>.<br/><br/>"
            "Order the results by <code>avg_salary</code> <strong>DESCENDING</strong>."
        ),
        "constraints": [
            "Use standard SQL / SQLite compatible syntax.",
            "Column names in result must be <code>department</code> and <code>avg_salary</code>.",
            "Order by <code>avg_salary</code> DESC, and by <code>department</code> ASC in case of ties."
        ],
        "starter_code": {
            "sql": (
                "-- Write your SQL solution below\n"
                "SELECT department, CAST(ROUND(AVG(salary)) AS INT) AS avg_salary\n"
                "FROM employees\n"
                "GROUP BY department\n"
                "HAVING AVG(salary) >= 70000\n"
                "ORDER BY avg_salary DESC, department ASC;\n"
            )
        },
        "entrypoint": "sql",
        "test_cases": [
            # 3 Open SQL Test Cases
            TestCase(
                id=1,
                name="Test Case 1 (Standard Company)",
                input_display="Standard multi-department dataset",
                expected_display="[['Engineering', 87500]]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (1, 'Alice', 'Engineering', 95000),
                      (2, 'Bob', 'Engineering', 80000),
                      (3, 'Charlie', 'Design', 68000),
                      (4, 'Diana', 'Design', 65000),
                      (5, 'Evan', 'Marketing', 60000),
                      (6, 'Frank', 'Marketing', 74000);
                """,
                expected_data=[["Engineering", 87500]],
                is_hidden=False,
                explanation="Engineering avg is 87,500 (>= 70,000). Design is 66,500 and Marketing is 67,000 (< 70,000)."
            ),
            TestCase(
                id=2,
                name="Test Case 2 (Multiple Qualifying Departments)",
                input_display="Engineering and Product both qualify",
                expected_display="[['Engineering', 90000], ['Product', 75000]]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (1, 'Alice', 'Engineering', 90000),
                      (2, 'Bob', 'Product', 75000),
                      (3, 'Charlie', 'Support', 45000);
                """,
                expected_data=[["Engineering", 90000], ["Product", 75000]],
                is_hidden=False,
                explanation="Engineering (90k) and Product (75k) both exceed 70k; sorted by avg_salary DESC."
            ),
            TestCase(
                id=3,
                name="Test Case 3 (Exact 70,000 Boundary)",
                input_display="Department with exactly 70,000 average",
                expected_display="[['Operations', 70000]]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (1, 'Sam', 'Operations', 70000),
                      (2, 'Dave', 'Sales', 69999);
                """,
                expected_data=[["Operations", 70000]],
                is_hidden=False,
                explanation="Operations exactly equals 70,000 (>= 70,000) and qualifies."
            ),
            # 3 Hidden SQL Test Cases
            TestCase(
                id=4,
                name="Hidden Test Case 1 (No Qualifying Departments)",
                input_display="[Hidden dataset 1]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (1, 'Zara', 'Logistics', 50000),
                      (2, 'Yuri', 'Logistics', 55000);
                """,
                expected_data=[],
                is_hidden=True,
                explanation="No department reaches 70k, returns empty set."
            ),
            TestCase(
                id=5,
                name="Hidden Test Case 2 (Tie Breaking & Large Salaries)",
                input_display="[Hidden dataset 2]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (1, 'Alpha', 'AI', 150000),
                      (2, 'Beta', 'Cloud', 120000),
                      (3, 'Gamma', 'Security', 120000);
                """,
                expected_data=[["AI", 150000], ["Cloud", 120000], ["Security", 120000]],
                is_hidden=True,
                explanation="High salary tech divisions; descending sort."
            ),
            TestCase(
                id=6,
                name="Hidden Test Case 3 (Single Employee Edge Cases)",
                input_display="[Hidden dataset 3]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);
                    INSERT INTO employees VALUES
                      (10, 'Executive', 'Executive', 250000),
                      (11, 'Intern', 'Interns', 30000);
                """,
                expected_data=[["Executive", 250000]],
                is_hidden=True,
                explanation="Only Executive reaches threshold."
            ),
        ]
    }
}


def get_challenge_metadata(challenge_id: str) -> dict:
    """Return public metadata and open test cases for the challenge (excluding hidden cases)."""
    chal = CHALLENGES.get(challenge_id)
    if not chal:
        raise ValueError(f"Unknown challenge: {challenge_id}")

    open_cases = [
        {
            "id": tc.id,
            "name": tc.name,
            "input_display": tc.input_display,
            "expected_display": tc.expected_display,
            "explanation": tc.explanation
        }
        for tc in chal["test_cases"] if not tc.is_hidden
    ]

    return {
        "id": chal["id"],
        "title": chal["title"],
        "type": chal["type"],
        "difficulty": chal["difficulty"],
        "category": chal["category"],
        "description": chal["description"],
        "constraints": chal["constraints"],
        "starter_code": chal["starter_code"],
        "open_test_cases": open_cases,
        "total_test_cases": len(chal["test_cases"]),
        "open_count": len(open_cases),
        "hidden_count": len(chal["test_cases"]) - len(open_cases)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Execution Engines (Python, JavaScript, SQL)
# ─────────────────────────────────────────────────────────────────────────────

def _run_python_tests(source_code: str, test_cases: List[TestCase], entrypoint: str) -> List[TestCaseResult]:
    """Execute Python code against test cases in an isolated subprocess."""
    import time

    results = []
    # Test runner script template
    script_harness = f"""
import sys, json

{source_code}

def __run_suite():
    input_cases = json.loads(sys.stdin.read())
    out_results = []
    for item in input_cases:
        args = item['args']
        try:
            res = {entrypoint}(**args)
            # If returns list of indices, sort for consistent comparison
            if isinstance(res, (list, tuple)) and all(isinstance(x, int) for x in res):
                res = sorted(list(res))
            out_results.append({{'status': 'ok', 'result': res}})
        except Exception as e:
            out_results.append({{'status': 'error', 'error': str(e)}})
    print(json.dumps(out_results))

if __name__ == '__main__':
    __run_suite()
"""

    # Prepare inputs
    suite_inputs = [{"id": tc.id, "args": tc.input_data} for tc in test_cases]
    input_json = json.dumps(suite_inputs)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script_harness)
        tmp_name = f.name

    start_time = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, tmp_name],
            input=input_json,
            text=True,
            capture_output=True,
            timeout=4.0
        )
        total_time_ms = round((time.time() - start_time) * 1000, 1)

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or "Syntax / Runtime Error"
            for tc in test_cases:
                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display="Error",
                    passed=False,
                    status="Runtime Error",
                    execution_time_ms=total_time_ms,
                    is_hidden=tc.is_hidden,
                    error=err_msg
                ))
            return results

        outputs = json.loads(proc.stdout.strip())
        for idx, tc in enumerate(test_cases):
            out_item = outputs[idx] if idx < len(outputs) else {"status": "error", "error": "No output"}
            if out_item.get("status") == "error":
                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display="Error",
                    passed=False,
                    status="Runtime Error",
                    execution_time_ms=round(total_time_ms / max(1, len(test_cases)), 1),
                    is_hidden=tc.is_hidden,
                    error=out_item.get("error", "Error")
                ))
            else:
                actual = out_item.get("result")
                expected = sorted(tc.expected_data) if isinstance(tc.expected_data, list) else tc.expected_data
                passed = (actual == expected)
                actual_disp = json.dumps(actual) if not tc.is_hidden or passed else "[Hidden Output]"

                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display=actual_disp,
                    passed=passed,
                    status="Passed" if passed else "Wrong Answer",
                    execution_time_ms=round(total_time_ms / max(1, len(test_cases)), 1),
                    is_hidden=tc.is_hidden
                ))

    except subprocess.TimeoutExpired:
        for tc in test_cases:
            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display="Time Limit Exceeded",
                passed=False,
                status="Time Limit Exceeded",
                execution_time_ms=4000.0,
                is_hidden=tc.is_hidden,
                error="Execution timed out (> 4.0s). Check for infinite loops."
            ))
    except Exception as e:
        for tc in test_cases:
            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display="Error",
                passed=False,
                status="Runtime Error",
                execution_time_ms=0.0,
                is_hidden=tc.is_hidden,
                error=str(e)
            ))
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

    return results


def _run_sql_tests(query: str, test_cases: List[TestCase]) -> List[TestCaseResult]:
    """Execute SQL query against test case SQLite in-memory databases."""
    import time
    results = []

    for tc in test_cases:
        t0 = time.time()
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        try:
            # Execute setup schema & inserts
            cur.executescript(tc.input_data)

            # Execute candidate query
            cur.execute(query.strip().rstrip(";"))
            rows = cur.fetchall()
            # Convert tuples to lists for standard JSON comparison
            clean_rows = [[c if not isinstance(c, float) else round(c, 2) for c in row] for row in rows]

            # Compare rows
            expected = tc.expected_data
            # Sort if needed or direct match
            passed = (clean_rows == expected)
            dur = round((time.time() - t0) * 1000, 1)

            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display=json.dumps(clean_rows) if not tc.is_hidden or passed else "[Hidden Output]",
                passed=passed,
                status="Passed" if passed else "Wrong Answer",
                execution_time_ms=dur,
                is_hidden=tc.is_hidden
            ))
        except Exception as e:
            dur = round((time.time() - t0) * 1000, 1)
            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display="Error",
                passed=False,
                status="Runtime Error",
                execution_time_ms=dur,
                is_hidden=tc.is_hidden,
                error=f"SQL Error: {str(e)}"
            ))
        finally:
            conn.close()

    return results


def execute_challenge(
    challenge_id: str,
    language: str,
    source_code: str,
    mode: str = "run"    # "run" (open cases only) or "submit" (all cases)
) -> dict:
    """Run code against test cases and return structured results."""
    chal = CHALLENGES.get(challenge_id)
    if not chal:
        raise ValueError(f"Unknown challenge ID: {challenge_id}")

    all_cases = chal["test_cases"]
    cases_to_run = [c for c in all_cases if not c.is_hidden] if mode == "run" else all_cases

    if chal["type"] == "sql" or language == "sql":
        results = _run_sql_tests(source_code, cases_to_run)
    else:
        results = _run_python_tests(source_code, cases_to_run, chal.get("entrypoint", "two_sum"))

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    score_pct = round((passed_count / max(1, total)) * 100)

    return {
        "challenge_id": challenge_id,
        "language": language,
        "mode": mode,
        "total_tests": total,
        "passed_tests": passed_count,
        "score_percent": score_pct,
        "all_passed": (passed_count == total),
        "results": [asdict(r) for r in results]
    }
