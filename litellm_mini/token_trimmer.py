"""
LiteLLM internal token trimmer.

Used by the router when a provider (e.g. Nexus) does not support
context_management natively.

Token counting: 1 token ≈ 4 UTF-8 characters.
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

    If already under threshold, returns the original list unchanged.
    System messages are always preserved.
    The most-recent half of non-system messages are always preserved.
    Oldest non-system messages are dropped first.
    Never mutates the input.
    """
    if count_tokens(messages) <= compact_threshold:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_sys = [m for m in messages if m.get("role") != "system"]
    keep_last_n = max(1, int(len(non_sys) * KEEP_RECENT_FRACTION))
    droppable = list(non_sys[:-keep_last_n])
    recent = non_sys[-keep_last_n:]

    while droppable and count_tokens(system + droppable + recent) > compact_threshold:
        droppable.pop(0)

    return system + droppable + recent
