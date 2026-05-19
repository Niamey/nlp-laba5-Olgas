"""Тести розширених метрик."""
from agent.extended_metrics import compute_extended_metrics, compute_report_quality


def test_report_completeness_sections():
    report = (
        "**Summary**\nOne line.\n\n"
        "**Analysis**\nDetails here.\n\n"
        "**Recommendation**\nDo something.\n\n"
        "**Labels**\nbug"
    )
    rq = compute_report_quality(report)
    assert rq["report_completeness_score"] == 1.0
    assert rq["sections_found"]["summary"] is True


def test_extended_metrics_minimal():
    state = {
        "loop_count": 0,
        "error_count": 0,
        "tool_results": [{"tool": "fetch_github_issue", "args": {"url": "x"}, "result": {"number": 1}}],
        "triage_findings": {"classification": {"type": "bug", "confidence": "high"}},
    }
    trajectory = {
        "grounding_passed": True,
        "fact_check": {"passed": True, "invented": [], "total_facts": 0},
        "tool_calls": 1,
        "duration_s": 10.0,
        "token_usage_est": {"input_tokens_est": 100, "output_tokens_est": 50},
        "tool_results": state["tool_results"],
    }
    report = "**Summary**\nx\n**Analysis**\ny\n**Recommendation**\nWe recommend closing."
    m = compute_extended_metrics(report=report, state=state, trajectory=trajectory)
    assert m["report_quality"]["report_word_count"] > 0
    assert m["classification"]["predicted_type"] == "bug"
    assert m["hallucination"]["grounding_passed"] is True


def test_min_search_rubric_check():
    task = {
        "task_id": "T002",
        "success_criteria": {"min_search_queries": 2, "must_search": True},
    }
    state = {"tool_results": [
        {"tool": "search_similar_issues", "args": {"query": "a"}, "result": []},
        {"tool": "search_similar_issues", "args": {"query": "b"}, "result": []},
    ]}
    m = compute_extended_metrics(report="test", state=state, trajectory={"tool_results": state["tool_results"]}, task=task)
    chk = m["rubric_autocheck"]["checks"]["min_search_queries"]
    assert chk["passed"] is True
    assert chk["actual"] == 2
