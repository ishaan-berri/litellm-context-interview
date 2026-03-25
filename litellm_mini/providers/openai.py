"""
OpenAI provider — context_management is native, passes through unchanged.
"""
from typing import Any, Dict, List

from litellm_mini.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["temperature", "max_tokens", "stream", "stop", "context_management"]

    def map_openai_params(
        self, non_default_params: Dict[str, Any], optional_params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        optional_params.update(non_default_params)
        return optional_params
