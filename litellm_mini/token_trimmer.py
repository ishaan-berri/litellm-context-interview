"""
LiteLLM internal token trimmer.

Used by the router when a provider (e.g. Nexus) does not support
context_management natively.

Token counting
--------------
Two counters are available:

count_tokens(messages)
    Fast approximation: 1 token ≈ 4 UTF-8 characters.
    Used inside trim_messages() — good enough for threshold comparisons.

count_tokens_tiktoken(messages)
    Accurate BPE count via a tiktoken-style encoder.
    More precise but noticeably slower: the encoder must be initialised
    once per process and then encode every character individually.
    P2: swap this in for count_tokens() once we benchmark the overhead
    against real traffic and confirm it stays within our latency budget.
"""
from typing import Any, Dict, List, Optional

from litellm_mini.types import ContextManagementEntry

CHARS_PER_TOKEN: int = 4
KEEP_RECENT_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Fast approximation — used by trim_messages
# ---------------------------------------------------------------------------

def count_tokens_in_message(message: Dict[str, Any]) -> int:
    content = message.get("content") or ""
    return max(1, len(content) // CHARS_PER_TOKEN)


def count_tokens(messages: List[Dict[str, Any]]) -> int:
    """Fast character-based approximation. 1 token ≈ 4 chars."""
    return sum(count_tokens_in_message(m) for m in messages)


# ---------------------------------------------------------------------------
# Accurate counter — P2, known to be slow
# ---------------------------------------------------------------------------

class _FakeTiktokenEncoder:
    """
    Stand-in for tiktoken.Encoding.

    A real encoder (e.g. tiktoken.get_encoding("cl100k_base")) loads a
    ~1 MB vocabulary file on first call and then BPE-encodes every token
    character by character.  We simulate that per-character work here so
    the slowness shows up in profiling even without a tiktoken dependency.

    P2 note: in production replace this class with:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    and call _encoder.encode(text) directly.
    """

    def encode(self, text: str) -> List[int]:
        # Simulates BPE: iterate every character, merge common bigrams.
        # This is intentionally O(n²) to surface the latency in benchmarks.
        tokens: List[int] = [ord(c) for c in text]
        merged: List[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] + tokens[i + 1] < 200:
                merged.append((tokens[i] + tokens[i + 1]) // 2)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged


_encoder = _FakeTiktokenEncoder()


def count_tokens_tiktoken(messages: List[Dict[str, Any]]) -> int:
    """
    Accurate token count using a tiktoken-style BPE encoder.

    More precise than count_tokens() but slower — the encoder processes
    every character individually.  Do not call this in the hot path until
    the P2 latency work is done.
    """
    total = 0
    for message in messages:
        content = message.get("content") or ""
        total += len(_encoder.encode(content))
        total += 4  # role + message overhead (mirrors tiktoken's chat format)
    return total


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
