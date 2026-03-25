"""
Nexus provider — no API-side context management support.

Nexus does not accept a context_management parameter.  When callers pass
context_management, the router uses LiteLLM's internal token trimmer
(token_trimmer.trim_messages) to reduce the message list before sending here.

Nexus never sees context_management in its request.
"""
from typing import Any, Dict, List

from litellm_mini.base_provider import BaseProvider


class NexusProvider(BaseProvider):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["temperature", "max_tokens", "stream", "stop"]
        # context_management is intentionally absent

    def map_openai_params(
        self, non_default_params: Dict[str, Any], optional_params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_completion_tokens"] = value
            else:
                optional_params[param] = value
        return optional_params
