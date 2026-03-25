"""
Anthropic provider — translates OpenAI context_management to Anthropic's edits format.

OpenAI:    [{"type": "compaction", "compact_threshold": 200000}]
Anthropic: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
           + header: anthropic-beta: compact-2026-01-12
"""
from typing import Any, Dict, List, Optional, Union

from litellm_mini.base_provider import BaseProvider

ANTHROPIC_TRIGGER_RATIO = 0.75


class AnthropicProvider(BaseProvider):
    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["temperature", "max_tokens", "stream", "stop", "context_management"]

    def map_openai_params(
        self, non_default_params: Dict[str, Any], optional_params: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        for param, value in non_default_params.items():
            if param == "context_management":
                translated = self._translate_context_management(value)
                if translated:
                    optional_params["context_management"] = translated
            else:
                optional_params[param] = value
        return optional_params

    def _translate_context_management(
        self, cm: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if isinstance(cm, dict) and "edits" in cm:
            return cm  # already Anthropic format
        if not isinstance(cm, list):
            return None
        edits = []
        for entry in cm:
            if entry.get("type") == "compaction":
                edit: Dict[str, Any] = {"type": "compact_20260112"}
                threshold = entry.get("compact_threshold")
                if threshold:
                    edit["trigger"] = {"type": "input_tokens", "value": int(threshold * ANTHROPIC_TRIGGER_RATIO)}
                edits.append(edit)
        return {"edits": edits} if edits else None
