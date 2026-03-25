from typing import Any, Dict, List, Literal, Optional, TypedDict


class ContextManagementEntry(TypedDict, total=False):
    type: str
    compact_threshold: int


class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant"]
    content: str
