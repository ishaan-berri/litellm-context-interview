"""
Starter test for Nexus context_management support.

Run with:
    pytest tests/test_nexus_context_management.py -v
"""
from litellm_mini.router import completion
from litellm_mini.token_trimmer import count_tokens


def test_nexus_context_management():
    big_messages = [{"role": "user", "content": "a" * 400}] * 20  # 2000 tokens

    result = completion(
        model="nexus/fast-v1",
        messages=big_messages,
        context_management=[{"type": "compaction", "compact_threshold": 500}],
    )

    assert count_tokens(result["messages"]) <= 500
