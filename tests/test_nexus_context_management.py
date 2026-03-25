"""
Tests for Nexus context_management support.

These tests are FAILING because of the TODO in router.py.

Your task: replace the `pass` in router.completion() with ~3 lines that call
get_compact_threshold() and trim_messages() from litellm_mini.token_trimmer.

Run with:
    pytest tests/test_nexus_context_management.py -v
"""
from litellm_mini.token_trimmer import count_tokens


def test_nexus_receives_trimmed_messages():
    """Messages must be trimmed to fit within compact_threshold."""
    from litellm_mini.router import completion

    threshold = 500
    big_messages = [{"role": "user", "content": "a" * 400}] * 20  # 2000 tokens

    result = completion(
        model="nexus/fast-v1",
        messages=big_messages,
        context_management=[{"type": "compaction", "compact_threshold": threshold}],
    )

    assert result["provider"] == "NexusProvider"
    assert count_tokens(result["messages"]) <= threshold


def test_nexus_does_not_receive_context_management_param():
    """context_management must NOT appear in the request sent to Nexus."""
    from litellm_mini.router import completion

    result = completion(
        model="nexus/fast-v1",
        messages=[{"role": "user", "content": "a" * 400}] * 20,
        context_management=[{"type": "compaction", "compact_threshold": 500}],
    )

    assert "context_management" not in result


def test_nexus_system_message_preserved():
    """System messages must survive trimming."""
    from litellm_mini.router import completion

    messages = (
        [{"role": "system", "content": "You are helpful."}]
        + [{"role": "user", "content": "a" * 400}] * 20
    )

    result = completion(
        model="nexus/fast-v1",
        messages=messages,
        context_management=[{"type": "compaction", "compact_threshold": 500}],
    )

    assert any(m["role"] == "system" for m in result["messages"])


def test_nexus_unknown_type_no_trim():
    """Unknown context_management type → no threshold → messages unchanged."""
    from litellm_mini.router import completion

    messages = [{"role": "user", "content": "hello"}] * 5

    result = completion(
        model="nexus/fast-v1",
        messages=messages,
        context_management=[{"type": "future_feature"}],
    )

    assert len(result["messages"]) == 5


def test_nexus_no_context_management_unchanged():
    """Without context_management, all messages pass through."""
    from litellm_mini.router import completion

    messages = [{"role": "user", "content": "hello"}] * 5
    result = completion(model="nexus/fast-v1", messages=messages)
    assert len(result["messages"]) == 5
