"""
Tests for Nexus context_management support via LiteLLM's token trimmer.

These tests are FAILING.  Your job is to implement trim_messages() in:
    litellm_mini/token_trimmer.py

Run with:
    pytest tests/test_nexus_context_management.py -v
"""
from litellm_mini.token_trimmer import (
    trim_messages,
    count_tokens,
    get_compact_threshold,
)


def msgs(*pairs) -> list[dict]:
    return [{"role": r, "content": c} for r, c in pairs]


# ---------------------------------------------------------------------------
# get_compact_threshold (already implemented — confirms expected shape)
# ---------------------------------------------------------------------------

def test_get_threshold_returns_value():
    assert get_compact_threshold([{"type": "compaction", "compact_threshold": 200000}]) == 200000


def test_get_threshold_none_for_unknown_type():
    assert get_compact_threshold([{"type": "future_feature"}]) is None


def test_get_threshold_none_for_no_config():
    assert get_compact_threshold(None) is None


# ---------------------------------------------------------------------------
# trim_messages — no-op cases
# ---------------------------------------------------------------------------

def test_no_trim_when_under_threshold():
    messages = msgs(("user", "a" * 40))   # 10 tokens
    assert trim_messages(messages, compact_threshold=1000) is messages  # same object


def test_no_trim_when_exactly_at_threshold():
    messages = msgs(("user", "a" * 400))  # 100 tokens
    assert trim_messages(messages, compact_threshold=100) is messages


# ---------------------------------------------------------------------------
# trim_messages — trimming behaviour
# ---------------------------------------------------------------------------

def test_result_fits_within_threshold():
    messages = msgs(*[("user", "a" * 400)] * 20)   # 2000 tokens
    result = trim_messages(messages, compact_threshold=500)
    assert count_tokens(result) <= 500


def test_does_not_mutate_input():
    messages = msgs(*[("user", "a" * 400)] * 10)
    trim_messages(messages, compact_threshold=200)
    assert len(messages) == 10


def test_system_messages_always_kept():
    system = {"role": "system", "content": "You are helpful."}
    messages = [system] + msgs(*[("user", "a" * 400)] * 20)
    result = trim_messages(messages, compact_threshold=500)
    assert any(m["role"] == "system" for m in result)


def test_recent_messages_kept():
    messages = (
        msgs(*[("user", f"old {i}") for i in range(18)])
        + msgs(("user", "RECENT_A"), ("user", "RECENT_B"))
    )
    result = trim_messages(messages, compact_threshold=30)
    assert any(m["content"] == "RECENT_B" for m in result)


def test_oldest_dropped_first():
    messages = msgs(("user", "OLDEST"), ("user", "a" * 400), ("user", "a" * 400), ("user", "NEWEST"))
    result = trim_messages(messages, compact_threshold=250)
    contents = [m["content"] for m in result]
    assert "NEWEST" in contents
    assert "OLDEST" not in contents


def test_single_message_always_returned():
    messages = msgs(("user", "a" * 4000))   # 1000 tokens, way over threshold
    result = trim_messages(messages, compact_threshold=10)
    assert len(result) >= 1


def test_output_is_integers():
    """count_tokens must always return int, not float."""
    messages = msgs(("user", "a" * 401))
    assert isinstance(count_tokens(messages), int)


# ---------------------------------------------------------------------------
# End-to-end via router
# ---------------------------------------------------------------------------

def test_nexus_receives_trimmed_messages():
    from litellm_mini.router import completion

    threshold = 500
    big_messages = [{"role": "user", "content": "a" * 400}] * 20   # 2000 tokens

    result = completion(
        model="nexus/fast-v1",
        messages=big_messages,
        context_management=[{"type": "compaction", "compact_threshold": threshold}],
    )

    assert result["provider"] == "NexusProvider"
    assert count_tokens(result["messages"]) <= threshold


def test_nexus_does_not_receive_context_management_param():
    from litellm_mini.router import completion

    result = completion(
        model="nexus/fast-v1",
        messages=[{"role": "user", "content": "a" * 400}] * 20,
        context_management=[{"type": "compaction", "compact_threshold": 500}],
    )
    # context_management must NOT appear in the request sent to Nexus
    assert "context_management" not in result


def test_nexus_unknown_context_management_type_no_trim():
    """Unknown type → no threshold → messages pass through unchanged."""
    from litellm_mini.router import completion

    messages = [{"role": "user", "content": "hello"}] * 5
    result = completion(
        model="nexus/fast-v1",
        messages=messages,
        context_management=[{"type": "future_feature"}],
    )
    assert len(result["messages"]) == 5
