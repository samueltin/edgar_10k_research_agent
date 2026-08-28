"""Tests for segment_extractor_node.

Two real bugs are covered here, both found from real hardware/real filing
testing, not designed speculatively upfront:

1. The provider-aware context limit (test_uses_provider_aware_context_limit),
   fixing a real 4096-token Ollama context overflow.

2. find_segment_table_by_total_revenue() (most of this file), fixing a
   fabricated segment number found on a real MSFT run, and then validated
   against real MD&A text from FOUR companies (MSFT, GOOG, AMZN, ORCL) plus
   IBM's "incorporated by reference" stub case -- an initial MSFT-only
   heading-keyword fix was tried and found to NOT generalize at all once
   tested against the other three companies' real text.
"""


class _FakeSegment:
    def __init__(self, segment_name, revenue):
        self.segment_name = segment_name
        self.revenue = revenue


class _FakeSegmentExtraction:
    def __init__(self, segments):
        self.segments = segments


class _FakeStructuredLLM:
    def __init__(self, segments_to_return):
        self._segments_to_return = segments_to_return

    def invoke(self, prompt):
        self.last_prompt = prompt
        return _FakeSegmentExtraction(self._segments_to_return)


class _FakeLLM:
    def __init__(self, segments_to_return):
        self._segments_to_return = segments_to_return
        self.structured_llm = None

    def with_structured_output(self, schema):
        self.structured_llm = _FakeStructuredLLM(self._segments_to_return)
        return self.structured_llm


def test_uses_provider_aware_context_limit(monkeypatch):
    """Confirms segment_extractor_node asks context_limits.py for the
    right limit rather than using a hardcoded constant -- the real bug
    found from a 4096-token Ollama context overflow."""
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    captured = {}

    def fake_get_max_input_chars(task):
        captured["task"] = task
        return 20

    fake_llm = _FakeLLM([_FakeSegment("Productivity", 1000.0)])

    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", fake_get_max_input_chars
    )
    monkeypatch.setattr("edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: fake_llm)

    long_text = "A" * 1000
    state = {"ticker": "TEST", "mda_text": long_text}
    segment_extractor_node(state)

    assert captured["task"] == "segment_extraction"
    assert "A" * 21 not in fake_llm.structured_llm.last_prompt


def test_returns_extracted_segments(monkeypatch):
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    fake_segments = [_FakeSegment("Productivity", 1000.0), _FakeSegment("Cloud", 2000.0)]
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", lambda task: 100_000
    )
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: _FakeLLM(fake_segments)
    )

    state = {"ticker": "TEST", "mda_text": "some MD&A text"}
    result = segment_extractor_node(state)

    assert result["extracted_segments"] == fake_segments
    assert result["validation_status"] == "PENDING"


# --- Real excerpts from four real companies' MD&A text ---
# Real bug found: MSFT's segment NAMES are defined ~7,000 characters before
# its real revenue table; the four companies below show FOUR DIFFERENT real
# lead-in conventions (only MSFT's contains the word "segment" at all).

REAL_MSFT_DEFINITION_EXCERPT = (
    "nancial performance based on the following three segments: Productivity "
    "and Business Processes, Intelligent Cloud, and More Personal Computing. "
    "The se"
)

REAL_MSFT_SEGMENT_TABLE = (
    "SEGMENT RESULTS OF OPERATIONS\n\n     (In millions, except percentages)\n\n"
    "     2026\n\n     2025\n\n     PercentageChange \n\n"
    "     Productivity and Business Processes\n\n     Revenue\n\n     $ \n\n"
    "     139,996\n\n     $\n\n     120,810\n\n     16%\n\n"
    "     Cost of revenue\n\n     25,017\n\n     22,422\n\n     12%\n\n"
    "     Operating expenses\n\n     31,100\n\n     28,615\n\n     9%\n\n"
    "     Operating income\n\n     $\n\n     83,879\n\n     $\n\n     69,773\n\n     20%\n\n"
    "     Intelligent Cloud\n\n     Revenue\n\n     $\n\n     137,791\n\n     $\n\n"
    "     106,265\n\n     30%\n\n"
    "     Cost of revenue\n\n     57,876\n\n     40,171\n\n     44%\n\n"
    "     Operating expenses\n\n     22,943\n\n     21,505\n\n     7%\n\n"
    "     Operating income\n\n     $\n\n     56,972\n\n     $\n\n     44,589\n\n     28%\n\n"
    "     More Personal Computing\n\n     Revenue\n\n     $\n\n     54,052\n\n     $\n\n"
    "     54,649\n\n     (1)%\n\n"
    "     Cost of revenue\n\n     23,481\n\n     25,238\n\n     (7)%\n\n"
    "     Operating expenses\n\n     16,185\n\n     15,245\n\n     6%\n\n"
    "     Operating income\n\n     $\n\n     14,386\n\n     $\n\n     14,166\n\n     2%\n\n"
    "     Total\n\n     Revenue\n\n     $\n\n     331,839\n\n     $\n\n     281,"
)

# Real excerpt: GOOG's revenue table has NO "segment" keyword at all --
# "The following table presents revenues by type", a completely different
# convention from MSFT's, in a different number format (packed against
# labels, not one number per line).
REAL_GOOG_EXCERPT = (
    "Financial Results\n\nRevenues\n\nThe following table presents revenues "
    "by type (in millions):\n\nYear Ended December 31,\n\n20242025\n\n"
    "Google Search & other$198,084\u00a0$224,532\u00a0\n\nYouTube ads36,147\u00a040,367\u00a0\n\n"
    "Google Network30,359\u00a029,792\u00a0\n\nGoogle advertising264,590\u00a0294,691\u00a0\n\n"
    "Google subscriptions, platforms, and devices\n\n40,340\u00a048,030\u00a0\n\n"
    "Google Services total304,930\u00a0342,721\u00a0\n\nGoogle Cloud43,229\u00a058,705\u00a0\n\n"
    "Other Bets1,648\u00a01,537\u00a0\n\nHedging gains (losses)211\u00a0(127)\n\n"
    "Total revenues$350,018\u00a0$402,836\u00a0\n\nGoogle Services\n\n"
)

# Real excerpt: AMZN's convention is different again -- "Net sales
# information is as follows", no "segment" keyword either.
REAL_AMZN_EXCERPT = (
    "Net sales information is as follows (in millions):\n\n"
    "Year Ended December 31,\n\n\u00a020242025\n\nNet Sales:\n\n"
    "North America$387,497\u00a0$426,305\u00a0\n\nInternational142,906\u00a0161,894\u00a0\n\n"
    "AWS107,556\u00a0128,725\u00a0\n\nConsolidated$637,959\u00a0$716,924\u00a0\n\n"
    "Year-over-year Percentage Growth:\n\nNorth America10\u00a0%10\u00a0%\n\n"
)

# Real excerpt: ORCL presents a GEOGRAPHIC breakdown ending in the SAME
# total revenue figure immediately before its real BUSINESS breakdown --
# the real ambiguity found that required the geography-exclusion signal,
# not just numeric density alone (both scored identically on density).
REAL_ORCL_EXCERPT = (
    "Total Revenues by Geography:\n\n     Americas\n\n     $\n\n     44,478\n\n"
    "     22%\n\n     22%\n\n     $\n\n     36,339\n\n     EMEA(1)\n\n     15,297\n\n"
    "     9%\n\n     3%\n\n     14,025\n\n     Asia Pacific\n\n     7,582\n\n"
    "     8%\n\n     8%\n\n     7,035\n\n     Total revenues\n\n     67,357\n\n"
    "     17%\n\n     16%\n\n     57,399\n\n     Total Operating Expenses\n\n"
    "     46,751\n\n     18%\n\n     17%\n\n     39,721\n\n     Total Operating Margin\n\n"
    "     $\n\n     20,606\n\n     17%\n\n     13%\n\n     $\n\n     17,678\n\n"
    "     Total Operating Margin %\n\n     31%\n\n     31%\n\n"
    "     % Revenues by Geography:\n\n     Americas\n\n     66%\n\n     63%\n\n"
    "     EMEA\n\n     23%\n\n     25%\n\n     Asia Pacific\n\n     11%\n\n     12%\n\n"
    "     Total Revenues by Business:\n\n     Cloud and software\n\n     $\n\n"
    "     58,530\n\n     19%\n\n     17%\n\n     $\n\n     49,230\n\n     Hardware\n\n"
    "     3,084\n\n     5%\n\n     3%\n\n     2,936\n\n     Services\n\n     5,743\n\n"
    "     10%\n\n     8%\n\n     5,233\n\n     Total revenues\n\n     $\n\n"
    "     67,357\n\n     17%\n\n     16%\n\n     $\n\n     57,399\n\n"
)

# Real excerpt: IBM's entire Item 7 is just this -- no numbers at all.
REAL_IBM_STUB = (
    "Item 7. Management\u2019s Discussion and Analysis of Financial Condition "
    "and Results of Operations:\n\nRefer to pages 6 through 38 of IBM\u2019s "
    "2025 Annual Report to Stockholders, which are incorporated herein by reference."
)


def test_finds_real_msft_table_anchored_on_known_total_revenue():
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = REAL_MSFT_DEFINITION_EXCERPT + ("X" * 5000) + REAL_MSFT_SEGMENT_TABLE
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=331_839_000_000.0)

    assert result is not None
    assert "139,996" in result
    assert "137,791" in result
    assert "54,052" in result
    assert "331,839" in result


def test_finds_real_google_table_despite_no_segment_keyword():
    """GOOG's real lead-in never says 'segment' at all -- confirms the
    total-revenue anchor works regardless of label wording, unlike the
    MSFT-only heading approach this replaced."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = ("Y" * 5000) + REAL_GOOG_EXCERPT
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=402_836_000_000.0)

    assert result is not None
    assert "304,930" in result   # Google Services total
    assert "58,705" in result    # Google Cloud
    assert "1,537" in result     # Other Bets
    assert "402,836" in result   # Total


def test_finds_real_amazon_table_despite_no_segment_keyword():
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = ("Y" * 5000) + REAL_AMZN_EXCERPT
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=716_924_000_000.0)

    assert result is not None
    assert "426,305" in result   # North America
    assert "161,894" in result   # International
    assert "128,725" in result   # AWS
    assert "716,924" in result   # Consolidated total


def test_excludes_geographic_breakdown_and_finds_real_oracle_business_table():
    """The real ambiguity found: Oracle's geography breakdown ends in the
    SAME total revenue figure as its real business breakdown, immediately
    before it. Numeric density alone ties (both score identically) --
    this confirms the geography-exclusion signal correctly picks the
    business breakdown instead."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = ("Y" * 5000) + REAL_ORCL_EXCERPT
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=67_357_000_000.0)

    assert result is not None
    assert "Total Revenues by Business" in result
    assert "58,530" in result   # Cloud and software
    assert "3,084" in result    # Hardware
    assert "5,743" in result    # Services
    # the geographic breakdown's own heading should NOT be what anchors the result
    assert "Americas" not in result.split("Total Revenues by Business")[0][-50:]


def test_boundary_clip_excludes_geography_table_dollar_figures():
    """REAL BUG FOUND on a live ORCL run: even after correctly choosing
    the business table over the geography one, the flat backward window
    was wide enough to bleed into the adjacent geography table anyway,
    and the model reported Americas/EMEA/Asia Pacific as fake segments.
    Confirms the boundary clip removes the geography table's real dollar
    figures (44,478 / 15,297 / 7,582) while keeping the full business
    table intact."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = ("Y" * 5000) + REAL_ORCL_EXCERPT
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=67_357_000_000.0)

    # the geography table's own dollar figures must be excluded
    assert "44,478" not in result
    assert "36,339" not in result
    # the business table's figures must still be fully present
    assert "58,530" in result
    assert "3,084" in result
    assert "5,743" in result
    assert "67,357" in result


def test_boundary_clip_does_not_affect_companies_without_an_adjacent_table():
    """Regression guard: the boundary clip must not shrink extraction for
    companies where the other candidate is just a distant narrative
    mention, not an adjacent table -- MSFT, GOOG, and AMZN's real text
    all have their two candidates far apart, well outside the normal
    2,000-char extraction window already."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    full_text = REAL_MSFT_DEFINITION_EXCERPT + ("X" * 5000) + REAL_MSFT_SEGMENT_TABLE
    result = find_segment_table_by_total_revenue(full_text, xbrl_total_revenue=331_839_000_000.0)

    assert "139,996" in result
    assert "137,791" in result
    assert "54,052" in result


def test_boundary_clip_handles_a_very_close_other_candidate_without_empty_result():
    """Edge case: if the nearest other candidate is extremely close to the
    chosen one, the clip must not produce an empty or negative-length
    extraction."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    # Two occurrences of the same total only a few characters apart
    text = "Some text 12,345 right here, and 12,345 again just after."
    result = find_segment_table_by_total_revenue(text, xbrl_total_revenue=12_345_000_000.0)

    assert result is not None
    assert len(result) > 0
    assert "12,345" in result


def test_ibm_stub_with_no_numbers_returns_none_not_an_error():
    """IBM's real Item 7 is just a ~30-word incorporation-by-reference
    pointer -- no special-case handling needed, the same total-revenue
    search naturally finds zero candidates since there are no numbers at
    all in the stub text."""
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    result = find_segment_table_by_total_revenue(REAL_IBM_STUB, xbrl_total_revenue=64_000_000_000.0)

    assert result is None


def test_returns_none_when_xbrl_total_revenue_is_missing():
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    result = find_segment_table_by_total_revenue(REAL_MSFT_SEGMENT_TABLE, xbrl_total_revenue=None)

    assert result is None


def test_returns_none_when_xbrl_total_revenue_is_zero():
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    result = find_segment_table_by_total_revenue(REAL_MSFT_SEGMENT_TABLE, xbrl_total_revenue=0.0)

    assert result is None


def test_returns_none_when_total_not_found_in_text():
    from edgar_research_agent.agent.nodes.segment_extractor import find_segment_table_by_total_revenue

    result = find_segment_table_by_total_revenue("Some text with no matching numbers at all.", xbrl_total_revenue=999_000_000.0)

    assert result is None


def test_segment_extractor_node_uses_the_located_table_not_the_full_text(monkeypatch):
    """End-to-end: confirms segment_extractor_node actually sends the
    located table to the LLM, not the full original text."""
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    fake_llm = _FakeLLM([_FakeSegment("Intelligent Cloud", 137791000000.0)])
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", lambda task: 100_000
    )
    monkeypatch.setattr("edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: fake_llm)

    full_text = REAL_MSFT_DEFINITION_EXCERPT + ("X" * 5000) + REAL_MSFT_SEGMENT_TABLE
    state = {"ticker": "MSFT", "mda_text": full_text, "xbrl_total_revenue": 331_839_000_000.0}
    segment_extractor_node(state)

    sent_prompt = fake_llm.structured_llm.last_prompt
    assert "137,791" in sent_prompt
    # Some filler naturally leaks into a fixed-size extraction window when
    # the real table (1,061 chars here) is shorter than the window (2,000
    # chars) -- expected and harmless. What matters is the bulk of the
    # 5,000-char filler between the definitions and the real table is
    # excluded, confirming the extraction is anchored near the real table,
    # not just returning the whole document.
    assert "X" * 2000 not in sent_prompt


def test_segment_extractor_node_falls_back_when_total_revenue_missing(monkeypatch):
    """Graceful fallback: when xbrl_total_revenue isn't in state (or no
    candidate is found), the node must still work using the original
    capped-full-text behavior, not crash or send nothing."""
    from edgar_research_agent.agent.nodes.segment_extractor import segment_extractor_node

    fake_llm = _FakeLLM([_FakeSegment("Some Segment", 1000.0)])
    monkeypatch.setattr(
        "edgar_research_agent.agent.nodes.segment_extractor.get_max_input_chars", lambda task: 100_000
    )
    monkeypatch.setattr("edgar_research_agent.agent.nodes.segment_extractor.get_llm", lambda: fake_llm)

    state = {"ticker": "TEST", "mda_text": "Some MD&A text with no segment table heading at all, just prose."}
    result = segment_extractor_node(state)

    assert len(result["extracted_segments"]) == 1
    assert result["extracted_segments"][0].segment_name == "Some Segment"
    assert "Some MD&A text" in fake_llm.structured_llm.last_prompt
