"""Code execution service for coding and SQL interview challenges.

Supports Python, JavaScript (Node.js), and SQL (in-memory SQLite).
Isolates execution with strict timeouts, separate process boundaries,
and deterministic evaluation of open and hidden test cases.
"""
from __future__ import annotations

import json
import os
import shutil
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
            ),
            "java": (
                "class Solution {\n"
                "    public int[] twoSum(int[] nums, int target) {\n"
                "        // Write your solution here\n"
                "        java.util.Map<Integer, Integer> seen = new java.util.HashMap<>();\n"
                "        for (int i = 0; i < nums.length; i++) {\n"
                "            int comp = target - nums[i];\n"
                "            if (seen.containsKey(comp)) {\n"
                "                return new int[]{seen.get(comp), i};\n"
                "            }\n"
                "            seen.put(nums[i], i);\n"
                "        }\n"
                "        return new int[]{};\n"
                "    }\n"
                "}\n"
            )
        },
        "entrypoint": "two_sum",
        "js_entrypoint": "twoSum",
        "java_method": "twoSum",
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
    },
    "valid_parentheses": {
        "id": "valid_parentheses",
        "title": "Valid Parentheses (Stack)",
        "type": "code",
        "difficulty": "Easy",
        "category": "Data Structures & Stacks",
        "description": (
            "Given a string <code>s</code> containing just the characters <code>'('</code>, <code>')'</code>, "
            "<code>'{'</code>, <code>'}'</code>, <code>'['</code> and <code>']'</code>, determine if the input string is valid.<br/><br/>"
            "An input string is valid if:<br/>"
            "1. Open brackets must be closed by the same type of brackets.<br/>"
            "2. Open brackets must be closed in the correct order.<br/>"
            "3. Every close bracket has a corresponding open bracket of the same type."
        ),
        "constraints": [
            "1 &le; s.length &le; 10<sup>4</sup>",
            "<code>s</code> consists of parentheses only: <code>'()[]{}'</code>."
        ],
        "starter_code": {
            "python": (
                "def is_valid(s: str) -> bool:\n"
                "    # Return True if s has valid balanced parentheses, else False\n"
                "    stack = []\n"
                "    mapping = {')': '(', '}': '{', ']': '['}\n"
                "    for char in s:\n"
                "        if char in mapping.values():\n"
                "            stack.append(char)\n"
                "        elif char in mapping:\n"
                "            if not stack or stack.pop() != mapping[char]:\n"
                "                return False\n"
                "        else:\n"
                "            return False\n"
                "    return len(stack) == 0\n"
            ),
            "javascript": (
                "function isValid(s) {\n"
                "    // Return true if s has valid balanced parentheses, else false\n"
                "    const stack = [];\n"
                "    const map = { ')': '(', '}': '{', ']': '[' };\n"
                "    for (const char of s) {\n"
                "        if (char === '(' || char === '{' || char === '[') {\n"
                "            stack.push(char);\n"
                "        } else if (map[char]) {\n"
                "            if (stack.pop() !== map[char]) return false;\n"
                "        }\n"
                "    }\n"
                "    return stack.length === 0;\n"
                "}\n"
            ),
            "java": (
                "class Solution {\n"
                "    public boolean isValid(String s) {\n"
                "        // Return true if s has valid balanced parentheses, else false\n"
                "        java.util.Stack<Character> stack = new java.util.Stack<>();\n"
                "        for (char c : s.toCharArray()) {\n"
                "            if (c == '(' || c == '{' || c == '[') {\n"
                "                stack.push(c);\n"
                "            } else {\n"
                "                if (stack.isEmpty()) return false;\n"
                "                char top = stack.pop();\n"
                "                if (c == ')' && top != '(') return false;\n"
                "                if (c == '}' && top != '{') return false;\n"
                "                if (c == ']' && top != '[') return false;\n"
                "            }\n"
                "        }\n"
                "        return stack.isEmpty();\n"
                "    }\n"
                "}\n"
            )
        },
        "entrypoint": "is_valid",
        "js_entrypoint": "isValid",
        "java_method": "isValid",
        "test_cases": [
            TestCase(
                id=1,
                name="Test Case 1 (Single Pair)",
                input_display="s = '()'",
                expected_display="true",
                input_data={"s": "()"},
                expected_data=True,
                is_hidden=False,
                explanation="Open bracket is immediately closed by the same type."
            ),
            TestCase(
                id=2,
                name="Test Case 2 (Multiple Sequential)",
                input_display="s = '()[]{}'",
                expected_display="true",
                input_data={"s": "()[]{}"},
                expected_data=True,
                is_hidden=False,
                explanation="All pairs match and are properly closed sequentially."
            ),
            TestCase(
                id=3,
                name="Test Case 3 (Mismatched Pair)",
                input_display="s = '(]'",
                expected_display="false",
                input_data={"s": "(]"},
                expected_data=False,
                is_hidden=False,
                explanation="Round bracket closed with square bracket."
            ),
            TestCase(
                id=4,
                name="Hidden Test Case 1 (Interleaved Improper)",
                input_display="[Hidden dataset 1]",
                expected_display="[Hidden]",
                input_data={"s": "([)]"},
                expected_data=False,
                is_hidden=True,
                explanation="Closing order is wrong: '[' is closed by ')'."
            ),
            TestCase(
                id=5,
                name="Hidden Test Case 2 (Nested Symmetrical)",
                input_display="[Hidden dataset 2]",
                expected_display="[Hidden]",
                input_data={"s": "{[]}"},
                expected_data=True,
                is_hidden=True,
                explanation="Proper nested closing of square inside curly brackets."
            ),
            TestCase(
                id=6,
                name="Hidden Test Case 3 (Single Close Bracket Edge)",
                input_display="[Hidden dataset 3]",
                expected_display="[Hidden]",
                input_data={"s": "]"},
                expected_data=False,
                is_hidden=True,
                explanation="Close bracket with empty stack returns False."
            ),
        ]
    },
    "sql_customer_orders_total": {
        "id": "sql_customer_orders_total",
        "title": "Customer Spending Total (SQL JOIN)",
        "type": "sql",
        "difficulty": "Easy-Medium",
        "category": "Database & SQL",
        "description": (
            "You are given two tables: <code>customers</code> and <code>orders</code> with schemas:<br/>"
            "<pre style='background:#F8FAFA;padding:0.6rem;border-radius:4px;margin:0.5rem 0;'>"
            "customers(id INT, name VARCHAR, city VARCHAR)\n"
            "orders(id INT, customer_id INT, amount INT, status VARCHAR)</pre>"
            "Write a SQL query that joins the tables and returns the customer <code>name</code>, <code>city</code>, "
            "and their total spending as <code>total_spent</code> (sum of order amounts) for all orders where <code>status = 'completed'</code>.<br/><br/>"
            "Only include customers whose total completed spending is <strong>greater than or equal to 150</strong>.<br/>"
            "Order results by <code>total_spent</code> <strong>DESCENDING</strong>, and then by <code>name</code> <strong>ASCENDING</strong>."
        ),
        "constraints": [
            "Use standard SQL / SQLite compatible syntax.",
            "Column names in result must be <code>name</code>, <code>city</code>, and <code>total_spent</code>.",
            "Order by <code>total_spent</code> DESC, then <code>name</code> ASC."
        ],
        "starter_code": {
            "sql": (
                "-- Write your SQL solution below\n"
                "SELECT c.name, c.city, SUM(o.amount) AS total_spent\n"
                "FROM customers c\n"
                "JOIN orders o ON c.id = o.customer_id\n"
                "WHERE o.status = 'completed'\n"
                "GROUP BY c.id, c.name, c.city\n"
                "HAVING SUM(o.amount) >= 150\n"
                "ORDER BY total_spent DESC, c.name ASC;\n"
            )
        },
        "entrypoint": "sql",
        "test_cases": [
            TestCase(
                id=1,
                name="Test Case 1 (Standard Active Orders)",
                input_display="Multi-order customer dataset",
                expected_display="[['Alice', 'New York', 250], ['Bob', 'San Francisco', 180]]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES
                      (1, 'Alice', 'New York'),
                      (2, 'Bob', 'San Francisco'),
                      (3, 'Charlie', 'Austin');
                    INSERT INTO orders VALUES
                      (101, 1, 150, 'completed'),
                      (102, 1, 100, 'completed'),
                      (103, 2, 180, 'completed'),
                      (104, 3, 90, 'completed');
                """,
                expected_data=[["Alice", "New York", 250], ["Bob", "San Francisco", 180]],
                is_hidden=False,
                explanation="Alice (250) and Bob (180) exceed 150. Charlie (90) is excluded."
            ),
            TestCase(
                id=2,
                name="Test Case 2 (Filtering Out Pending/Cancelled Orders)",
                input_display="Orders with non-completed statuses",
                expected_display="[['David', 'Seattle', 200]]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES
                      (1, 'David', 'Seattle'),
                      (2, 'Emma', 'Chicago');
                    INSERT INTO orders VALUES
                      (101, 1, 200, 'completed'),
                      (102, 2, 500, 'pending'),
                      (103, 2, 50, 'completed');
                """,
                expected_data=[["David", "Seattle", 200]],
                is_hidden=False,
                explanation="Emma's 500 order is pending, only 50 completed, so she is filtered out (< 150)."
            ),
            TestCase(
                id=3,
                name="Test Case 3 (Exact 150 Threshold)",
                input_display="Customer with exactly 150 completed spending",
                expected_display="[['Grace', 'Boston', 150]]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES
                      (1, 'Grace', 'Boston'),
                      (2, 'Henry', 'Denver');
                    INSERT INTO orders VALUES
                      (101, 1, 75, 'completed'),
                      (102, 1, 75, 'completed'),
                      (103, 2, 149, 'completed');
                """,
                expected_data=[["Grace", "Boston", 150]],
                is_hidden=False,
                explanation="Grace meets the >= 150 boundary exactly; Henry is at 149 and excluded."
            ),
            TestCase(
                id=4,
                name="Hidden Test Case 1 (Empty Result Set)",
                input_display="[Hidden dataset 1]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES (1, 'Ian', 'Miami');
                    INSERT INTO orders VALUES (101, 1, 100, 'cancelled');
                """,
                expected_data=[],
                is_hidden=True,
                explanation="No completed orders, empty result set."
            ),
            TestCase(
                id=5,
                name="Hidden Test Case 2 (Alphabetical Tie-Break)",
                input_display="[Hidden dataset 2: Equal spending]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES
                      (1, 'Zoe', 'Dallas'),
                      (2, 'Adam', 'Atlanta');
                    INSERT INTO orders VALUES
                      (101, 1, 300, 'completed'),
                      (102, 2, 300, 'completed');
                """,
                expected_data=[["Adam", "Atlanta", 300], ["Zoe", "Dallas", 300]],
                is_hidden=True,
                explanation="Adam and Zoe both spent 300; sorted alphabetically by name (Adam before Zoe)."
            ),
            TestCase(
                id=6,
                name="Hidden Test Case 3 (Multi-order Aggregation Edge)",
                input_display="[Hidden dataset 3]",
                expected_display="[Hidden]",
                input_data="""
                    CREATE TABLE customers (id INT, name TEXT, city TEXT);
                    CREATE TABLE orders (id INT, customer_id INT, amount INT, status TEXT);
                    INSERT INTO customers VALUES
                      (1, 'Maya', 'San Jose'),
                      (2, 'Liam', 'Portland');
                    INSERT INTO orders VALUES
                      (101, 1, 50, 'completed'),
                      (102, 1, 60, 'completed'),
                      (103, 1, 70, 'completed'),
                      (104, 2, 10, 'completed');
                """,
                expected_data=[["Maya", "San Jose", 180]],
                is_hidden=True,
                explanation="Maya has 50+60+70 = 180 >= 150."
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
            is_syntax = "SyntaxError" in err_msg or "IndentationError" in err_msg or "TabError" in err_msg
            status_label = "Compilation Error" if is_syntax else "Runtime Error"
            for tc in test_cases:
                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display="Error",
                    passed=False,
                    status=status_label,
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


def _run_javascript_tests(source_code: str, test_cases: List[TestCase], entrypoint: str) -> List[TestCaseResult]:
    """Execute JavaScript code (Node.js) against test cases in an isolated subprocess."""
    import time

    results = []
    script_harness = f"""
const fs = require('fs');

{source_code}

function __runSuite() {{
    const inputCases = JSON.parse(fs.readFileSync(0, 'utf-8'));
    const outResults = [];
    for (const item of inputCases) {{
        const args = item.args;
        try {{
            let res;
            if (Array.isArray(args)) {{
                res = {entrypoint}(...args);
            }} else if (typeof args === 'object' && args !== null) {{
                res = {entrypoint}(...Object.values(args));
            }} else {{
                res = {entrypoint}(args);
            }}
            if (Array.isArray(res) && res.every(x => typeof x === 'number')) {{
                res = res.slice().sort((a, b) => a - b);
            }}
            outResults.push({{ status: 'ok', result: res }});
        }} catch (e) {{
            outResults.push({{ status: 'error', error: e.message || String(e) }});
        }}
    }}
    console.log(JSON.stringify(outResults));
}}

__runSuite();
"""

    suite_inputs = [{"id": tc.id, "args": tc.input_data} for tc in test_cases]
    input_json = json.dumps(suite_inputs)

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script_harness)
        tmp_name = f.name

    start_time = time.time()
    try:
        proc = subprocess.run(
            ["node", tmp_name],
            input=input_json,
            text=True,
            capture_output=True,
            timeout=4.0
        )
        total_time_ms = round((time.time() - start_time) * 1000, 1)

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or "JavaScript Syntax / Runtime Error"
            is_syntax = "SyntaxError" in err_msg
            status_label = "Compilation Error" if is_syntax else "Runtime Error"
            for tc in test_cases:
                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display="Error",
                    passed=False,
                    status=status_label,
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


def _generate_java_call(challenge_id: str, method: str, tc_id: int, input_data: Any) -> str:
    if challenge_id == "two_sum":
        nums_str = ", ".join(str(x) for x in input_data["nums"])
        target = input_data["target"]
        return f"""
        try {{
            int[] res = sol.{method}(new int[]{{{nums_str}}}, {target});
            if (res != null) {{
                java.util.Arrays.sort(res);
                System.out.println("RESULT:{tc_id}:" + java.util.Arrays.toString(res));
            }} else {{
                System.out.println("RESULT:{tc_id}:null");
            }}
        }} catch (Throwable t) {{
            System.out.println("ERROR:{tc_id}:" + t.toString());
        }}
        """
    elif challenge_id == "valid_parentheses":
        escaped_s = input_data["s"].replace('"', '\\"')
        return f"""
        try {{
            boolean res = sol.{method}("{escaped_s}");
            System.out.println("RESULT:{tc_id}:" + res);
        }} catch (Throwable t) {{
            System.out.println("ERROR:{tc_id}:" + t.toString());
        }}
        """
    return f'System.out.println("ERROR:{tc_id}:Unsupported Java challenge");'


def _run_java_tests(challenge_id: str, source_code: str, test_cases: List[TestCase], method: str) -> List[TestCaseResult]:
    """Execute Java code against test cases in an isolated subprocess.
    
    Checks if javac/java are installed on the host system.
    If not installed, returns an informative runtime error prompting OpenJDK 17+ installation.
    If installed, compiles Solution.java + TestRunner.java and executes safely with timeouts.
    """
    import time
    javac_bin = shutil.which("javac")
    java_bin = shutil.which("java")

    if not javac_bin or not java_bin:
        results = []
        for tc in test_cases:
            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display="JDK Not Found",
                passed=False,
                status="Runtime Error",
                execution_time_ms=0.0,
                is_hidden=tc.is_hidden,
                error="Java execution requires a JDK (javac/java). Please install OpenJDK 17+ or switch language to Python / JavaScript."
            ))
        return results

    results = []
    test_calls = "\n".join([_generate_java_call(challenge_id, method, tc.id, tc.input_data) for tc in test_cases])

    runner_source = f"""
public class TestRunner {{
    public static void main(String[] args) {{
        Solution sol = new Solution();
{test_calls}
    }}
}}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sol_file = os.path.join(tmpdir, "Solution.java")
        runner_file = os.path.join(tmpdir, "TestRunner.java")

        with open(sol_file, "w", encoding="utf-8") as f:
            f.write(source_code)
        with open(runner_file, "w", encoding="utf-8") as f:
            f.write(runner_source)

        t0 = time.time()
        try:
            compile_proc = subprocess.run(
                [javac_bin, "Solution.java", "TestRunner.java"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=8.0
            )
            if compile_proc.returncode != 0:
                err_msg = compile_proc.stderr.strip() or "Java Compilation Error"
                for tc in test_cases:
                    results.append(TestCaseResult(
                        test_case_id=tc.id,
                        name=tc.name,
                        input_display=tc.input_display,
                        expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                        actual_display="Compilation Error",
                        passed=False,
                        status="Runtime Error",
                        execution_time_ms=round((time.time() - t0) * 1000, 1),
                        is_hidden=tc.is_hidden,
                        error=err_msg
                    ))
                return results
        except subprocess.TimeoutExpired:
            for tc in test_cases:
                results.append(TestCaseResult(
                    test_case_id=tc.id,
                    name=tc.name,
                    input_display=tc.input_display,
                    expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                    actual_display="Compilation Timeout",
                    passed=False,
                    status="Time Limit Exceeded",
                    execution_time_ms=8000.0,
                    is_hidden=tc.is_hidden,
                    error="javac compilation timed out (> 8.0s)"
                ))
            return results

        try:
            run_proc = subprocess.run(
                [java_bin, "-cp", ".", "TestRunner"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=5.0
            )
            total_time_ms = round((time.time() - t0) * 1000, 1)

            output_map = {}
            for line in run_proc.stdout.splitlines():
                if line.startswith("RESULT:"):
                    parts = line.split(":", 2)
                    if len(parts) == 3:
                        output_map[int(parts[1])] = ("result", parts[2])
                elif line.startswith("ERROR:"):
                    parts = line.split(":", 2)
                    if len(parts) == 3:
                        output_map[int(parts[1])] = ("error", parts[2])

            for tc in test_cases:
                item = output_map.get(tc.id, ("error", "No output generated"))
                if item[0] == "error":
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
                        error=item[1]
                    ))
                else:
                    actual_str = item[1].strip()
                    passed = False
                    if isinstance(tc.expected_data, bool):
                        passed = (actual_str.lower() == str(tc.expected_data).lower())
                    elif isinstance(tc.expected_data, list):
                        expected_str = str(sorted(tc.expected_data))
                        passed = (actual_str.replace(" ", "") == expected_str.replace(" ", ""))

                    disp = actual_str if not tc.is_hidden or passed else "[Hidden Output]"
                    results.append(TestCaseResult(
                        test_case_id=tc.id,
                        name=tc.name,
                        input_display=tc.input_display,
                        expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                        actual_display=disp,
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
                    execution_time_ms=5000.0,
                    is_hidden=tc.is_hidden,
                    error="Java execution timed out (> 5.0s). Check for infinite loops."
                ))

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
            err_str = str(e)
            is_syntax = "syntax error" in err_str.lower() or "near " in err_str.lower() or "unrecognized token" in err_str.lower()
            results.append(TestCaseResult(
                test_case_id=tc.id,
                name=tc.name,
                input_display=tc.input_display,
                expected_display="[Hidden]" if tc.is_hidden else tc.expected_display,
                actual_display="SQL Error",
                passed=False,
                status="SQL Syntax Error" if is_syntax else "Runtime Error",
                execution_time_ms=dur,
                is_hidden=tc.is_hidden,
                error=f"SQL Error: {err_str}"
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
    elif language.lower() in ("javascript", "js"):
        results = _run_javascript_tests(source_code, cases_to_run, chal.get("js_entrypoint", chal.get("entrypoint", "twoSum")))
    elif language.lower() in ("java",):
        results = _run_java_tests(chal["id"], source_code, cases_to_run, chal.get("java_method", "twoSum"))
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
