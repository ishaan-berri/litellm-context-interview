"""
Context management behaviour for providers that support it natively.
Already passing — do not modify.
"""
from litellm_mini.router import completion


def test_openai_forwards_context_management_unchanged():
    cm = [{"type": "compaction", "compact_threshold": 200000}]
    result = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        context_management=cm,
    )
    assert result["context_management"] == cm
    assert result["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_translates_to_edits():
    cm = [{"type": "compaction", "compact_threshold": 200000}]
    result = completion(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        context_management=cm,
    )
    assert "edits" in result["context_management"]
    assert result["context_management"]["edits"][0]["type"] == "compact_20260112"
    assert result["context_management"]["edits"][0]["trigger"]["value"] == 150000


def test_no_context_management_passes_all_messages():
    msgs = [{"role": "user", "content": "hi"}] * 5
    result = completion(model="nexus/fast-v1", messages=msgs)
    assert len(result["messages"]) == 5
