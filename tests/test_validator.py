"""Tests for the validation node's pass/fail routing."""
from edgar_research_agent.agent.nodes.validator import validator_node, route_validation
from edgar_research_agent.agent.schemas import SegmentKPI


def test_pass_within_threshold():
    state = {
        "xbrl_total_revenue": 1_000_000_000,
        "extracted_segments": [
            SegmentKPI(segment_name="Cloud", revenue=600_000_000),
            SegmentKPI(segment_name="Devices", revenue=399_000_000),
        ],
    }
    result = validator_node(state)
    assert result["validation_status"] == "PASS"


def test_fail_beyond_threshold():
    state = {
        "xbrl_total_revenue": 1_000_000_000,
        "extracted_segments": [
            SegmentKPI(segment_name="Cloud", revenue=600_000_000),
            SegmentKPI(segment_name="Devices", revenue=100_000_000),
        ],
    }
    result = validator_node(state)
    assert result["validation_status"] == "FAIL"
    assert "variance" in result["errors"]


def test_route_validation_directs_to_human_review_on_fail():
    state = {"validation_status": "FAIL"}
    assert route_validation(state) == "human_review"


def test_route_validation_ends_on_pass():
    state = {"validation_status": "PASS"}
    assert route_validation(state) == "end"
