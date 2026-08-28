"""Tests for the groundedness_checker_node, wired into the graph after
risk_summarizer_node. LLM-dependent check_groundedness is monkeypatched
so this stays offline and fast.
"""
from edgar_research_agent.agent.nodes.groundedness_checker import groundedness_checker_node
from edgar_research_agent.agent.groundedness import GroundednessResult


def test_produces_one_result_per_category_in_order(monkeypatch):
    calls = []

    def fake_check_groundedness(summary_text, source_text):
        calls.append((summary_text, source_text))
        return GroundednessResult(grounded=True, backend="custom_llm")

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.groundedness_checker.check_groundedness",
        fake_check_groundedness,
    )

    state = {
        "risk_summary_by_category": [
            {"heading": "General Risks", "summary": "summary one", "source_text": "source one"},
            {"heading": "Tax Risks", "summary": "summary two", "source_text": "source two"},
        ]
    }

    result = groundedness_checker_node(state)

    assert len(result["groundedness_results"]) == 2
    assert calls == [("summary one", "source one"), ("summary two", "source two")]


def test_empty_category_list_produces_empty_results(monkeypatch):
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.groundedness_checker.check_groundedness",
        lambda summary_text, source_text: GroundednessResult(grounded=True, backend="custom_llm"),
    )

    state = {"risk_summary_by_category": []}

    result = groundedness_checker_node(state)

    assert result["groundedness_results"] == []


def test_preserves_grounded_and_ungrounded_results_correctly(monkeypatch):
    results_by_summary = {
        "grounded summary": GroundednessResult(grounded=True, backend="custom_llm"),
        "ungrounded summary": GroundednessResult(
            grounded=False, backend="custom_llm",
            ungrounded_spans=[],
        ),
    }
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.groundedness_checker.check_groundedness",
        lambda summary_text, source_text: results_by_summary[summary_text],
    )

    state = {
        "risk_summary_by_category": [
            {"heading": "A", "summary": "grounded summary", "source_text": "x"},
            {"heading": "B", "summary": "ungrounded summary", "source_text": "y"},
        ]
    }

    result = groundedness_checker_node(state)

    assert result["groundedness_results"][0].grounded is True
    assert result["groundedness_results"][1].grounded is False
