"""Юніт-тести для agent.fact_extractor (regex fact-checker)."""
from __future__ import annotations

import pytest

from agent.fact_extractor import (
    extract_facts_from_report,
    check_facts_against_blob,
    format_fact_check_warning,
    FactCheckResult,
)


# ── Extraction ────────────────────────────────────────────────────

def test_extract_issue_numbers_hash():
    r = "This is a duplicate of #1234 and related to #56."
    f = extract_facts_from_report(r)
    assert "1234" in f["issue_numbers"]
    assert "56" in f["issue_numbers"]


def test_extract_issue_numbers_keyword():
    r = "See issue 4823 and PR #999"
    f = extract_facts_from_report(r)
    assert "4823" in f["issue_numbers"]
    assert "999" in f["issue_numbers"]


def test_skip_short_numbers():
    r = "Version 2.5 with 3 retries"  # лише 1-значні цифри в контексті
    f = extract_facts_from_report(r)
    assert "3" not in f["issue_numbers"]
    assert "2" not in f["issue_numbers"]


def test_extract_usernames():
    r = "Reported by @alice and @bob-smith, cc @charlie"
    f = extract_facts_from_report(r)
    assert "alice" in f["usernames"]
    assert "bob-smith" in f["usernames"]
    assert "charlie" in f["usernames"]


def test_username_ignore_generic():
    r = "@user reported this, @owner approved, @bot replied"
    f = extract_facts_from_report(r)
    assert f["usernames"] == []


def test_extract_file_paths():
    r = "Bug in src/auth/login.py and tests/test_handler.ts"
    f = extract_facts_from_report(r)
    assert any("login.py" in p for p in f["file_paths"])
    assert any("test_handler.ts" in p for p in f["file_paths"])


def test_extract_urls():
    r = "See https://github.com/foo/bar/issues/1 and http://example.com/path"
    f = extract_facts_from_report(r)
    assert any("github.com" in u for u in f["urls"])
    assert any("example.com" in u for u in f["urls"])


def test_extract_empty_report():
    f = extract_facts_from_report("")
    assert f == {"issue_numbers": [], "usernames": [], "file_paths": [], "urls": []}


def test_deduplication():
    r = "#42 again #42 once more #42"
    f = extract_facts_from_report(r)
    assert f["issue_numbers"].count("42") == 1


# ── check_facts_against_blob ─────────────────────────────────────

def test_no_facts_passes():
    fc = check_facts_against_blob("Generic summary without numbers.", "blob contents")
    assert fc.passed is True
    assert fc.total_facts == 0
    assert fc.facts_grounded_rate is None


def test_all_grounded_passes():
    report = "Issue #1234 by @alice was duplicate of #56."
    blob = "fetched_issue: number 1234; user alice; similar: 56"
    fc = check_facts_against_blob(report, blob)
    assert fc.passed is True
    assert fc.total_facts == 3
    assert fc.grounded_facts == 3
    assert fc.facts_grounded_rate == 1.0
    assert fc.invented == []


def test_some_invented_fails():
    report = "Issue #1234 by @ghost in src/fake/path.py"
    blob = "fetched_issue: number 1234; user alice"
    fc = check_facts_against_blob(report, blob, fail_threshold=0.8)
    assert fc.passed is False
    assert fc.facts_grounded_rate is not None
    assert fc.facts_grounded_rate < 0.8
    kinds = {x["kind"] for x in fc.invented}
    assert "usernames" in kinds
    assert "file_paths" in kinds


def test_path_basename_match_grounded():
    report = "Affects src/auth/login.py"
    blob = "tool result mentions login.py at root"
    fc = check_facts_against_blob(report, blob)
    assert fc.passed is True
    assert fc.invented == []


def test_threshold_boundary():
    # Витяг #N потребує мінімум 2 цифри (уникнення false positive з версіями як 2.5).
    report = "#101 #102 #103 #104 #105"
    blob = "issues 101, 102, 103, 104"
    fc = check_facts_against_blob(report, blob, fail_threshold=0.8)
    assert fc.total_facts == 5
    assert fc.grounded_facts == 4
    assert fc.facts_grounded_rate == 0.8
    assert fc.passed is True


def test_threshold_boundary_fail():
    report = "#101 #102 #103 #104 #105 #106"
    blob = "issues 101, 102, 103, 104"
    fc = check_facts_against_blob(report, blob, fail_threshold=0.8)
    assert fc.grounded_facts == 4
    assert fc.total_facts == 6
    assert fc.facts_grounded_rate is not None
    assert fc.facts_grounded_rate < 0.8
    assert fc.passed is False


def test_url_match():
    report = "More: https://github.com/foo/bar/issues/1234"
    blob = "see https://github.com/foo/bar/issues/1234 in fetch result"
    fc = check_facts_against_blob(report, blob)
    assert fc.passed is True


def test_url_mismatch():
    report = "Visit https://evil.example.com/steal"
    blob = "no urls here"
    fc = check_facts_against_blob(report, blob)
    assert fc.passed is False
    assert any(x["kind"] == "urls" for x in fc.invented)


# ── format helper ────────────────────────────────────────────────

def test_format_pass_no_facts():
    fc = FactCheckResult()
    s = format_fact_check_warning(fc)
    assert "no checkable facts" in s


def test_format_with_invented():
    fc = check_facts_against_blob(
        "Issue #999 by @nobody",
        "nothing relevant in blob",
    )
    s = format_fact_check_warning(fc)
    assert "FAIL" in s
    assert "INVENTED" in s
