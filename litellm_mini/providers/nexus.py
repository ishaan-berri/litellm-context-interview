"""
Nexus provider — no API-side context management support.

Nexus does not have a context_management API parameter.  Instead, LiteLLM
handles it client-side in transform_request(): the messages are trimmed using
the internal token trimmer before the request is sent.
"""
from typing import Any, Dict, List, Optional, Tuple

from litellm_mini.base_provider import BaseProvider
from litellm_mini.token_trimmer import get_compact_threshold, trim_messages
from litellm_mini.types import ContextManagementEntry


class NexusProvider(BaseProvider):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["temperature", "max_tokens", "stream", "stop"]
        # context_management intentionally absent — handled in transform_request

    def transform_request(
        self,
        messages: List[Dict[str, Any]],
        context_management: Optional[List[ContextManagementEntry]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[ContextManagementEntry]]]:
        """
        Trim messages to fit within compact_threshold when context_management
        is set, then drop context_management so it isn't forwarded to Nexus.

        TODO: implement this method.

        Steps:
          1. Call get_compact_threshold(context_management) to get the threshold.
          2. If a threshold is found, call trim_messages(messages, threshold).
          3. Return (trimmed_messages, None) — None signals the router to drop
             context_management from the outgoing request.
          4. If no threshold (unknown type or None), return (messages, None)
             unchanged — still drop context_management, Nexus doesn't know it.
        """
        raise NotImplementedError

    def map_openai_params(
        self, non_default_params: Dict[str, Any], optional_params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_completion_tokens"] = value
            else:
                optional_params[param] = value
        return optional_params
