"""
LiteLLM internal token trimmer.

Used by the router when a provider (e.g. Nexus) does not support
context_management natively.  The router calls trim_messages() to reduce
the conversation history before sending it to the provider.

Token counting
--------------
1 token ≈ 4 UTF-8 characters  (GPT-style approximation, no tiktoken required)

Trimming strategy
-----------------
When total token count exceeds compact_threshold:

1. System messages are ALWAYS kept — they carry the model's instructions.
2. Decide how many recent non-system messages to protect:
       keep_last_n = max(1, int(len(non_system) * KEEP_RECENT_FRACTION))
3. Split non-system into:
       droppable = messages older than the protected tail
       recent    = the last keep_last_n messages (never dropped)
4. Remove messages from the FRONT of droppable until
       count_tokens(system + remaining_droppable + recent) <= compact_threshold
5. If all droppable are exhausted and we're still over (e.g. huge system prompt),
   return system + recent anyway — never drop system or recent to hit a number.
"""
from typing import Any, Dict, List, Optional

from litellm_mini.types import ContextManagementEntry

CHARS_PER_TOKEN: int = 4
KEEP_RECENT_FRACTION: float = 0.5


def count_tokens_in_message(message: Dict[str, Any]) -> int:
    content = message.get("content") or ""
    return max(1, len(content) // CHARS_PER_TOKEN)


def count_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(count_tokens_in_message(m) for m in messages)


def get_compact_threshold(
    context_management: Optional[List[ContextManagementEntry]],
) -> Optional[int]:
    """Return the compact_threshold from the first compaction entry, or None."""
    if not context_management:
        return None
    for entry in context_management:
        if entry.get("type") == "compaction":
            return entry.get("compact_threshold")  # type: ignore[return-value]
    return None


def trim_messages(
    messages: List[Dict[str, Any]],
    compact_threshold: int,
) -> List[Dict[str, Any]]:
    """
    Return a trimmed copy of messages that fits within compact_threshold tokens.

    If already under the threshold, returns the original list unchanged.
    Never mutates the input list.

    See module docstring for the full trimming strategy.
    """
    # TODO: implement this
    raise NotImplementedError
