"""
Starter test for Nexus context_management support.

Run with:
    pytest tests/test_nexus_context_management.py -v
"""
from litellm_mini.router import completion


def test_nexus_context_management():
    big_messages = [{"role": "user", "content": "a" * 400}] * 20  # 2000 tokens

    result = completion(
        model="nexus/fast-v1",
        messages=big_messages,
        context_management=[{"type": "compaction", "compact_threshold": 1100}],
    )

    assert result["output"]["content"]
    assert result["usage"]["input_tokens"] >= 1
