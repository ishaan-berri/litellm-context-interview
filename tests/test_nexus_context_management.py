"""
Starter test for Nexus context_management support.

Run with:
    pytest tests/test_nexus_context_management.py -v
"""
import requests

from litellm_mini.router import completion
from litellm_mini.token_trimmer import count_tokens

NEXUS_URL = "https://fakecustomprovider-production.up.railway.app/v1/chat"
NEXUS_KEY = "nxk-test123"


def _call_nexus(messages: list, **kwargs) -> dict:
    resp = requests.post(
        NEXUS_URL,
        headers={"X-Nexus-Key": NEXUS_KEY, "Content-Type": "application/json"},
        json={"model": "nexus-ultra-v1", "messages": messages, **kwargs},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def test_nexus_context_management():
    big_messages = [{"role": "user", "content": "a" * 400}] * 20  # 2000 tokens

    # trim_messages should reduce this to fit under 500 tokens
    result = completion(
        model="nexus/fast-v1",
        messages=big_messages,
        context_management=[{"type": "compaction", "compact_threshold": 500}],
    )

    trimmed = result["messages"]
    assert count_tokens(trimmed) <= 500

    # send the trimmed messages to the live Nexus API
    response = _call_nexus(trimmed, max_completion_tokens=64)

    assert response["id"].startswith("nxresp-")
    assert response["output"]["content"]
    assert response["usage"]["input_tokens"] >= 1
