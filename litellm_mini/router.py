"""
Router: maps model names to providers, applies context_management.

Each provider declares how it handles context_management via transform_request():
  - OpenAI / Anthropic: return messages unchanged, keep context_management so
    map_openai_params can forward / translate it.
  - Nexus: trim messages client-side, return None for context_management so
    the param is never forwarded.
"""
from typing import Any, Dict, List, Optional

from litellm_mini.base_provider import BaseProvider
from litellm_mini.providers.anthropic import AnthropicProvider
from litellm_mini.providers.nexus import NexusProvider
from litellm_mini.providers.openai import OpenAIProvider
from litellm_mini.types import ContextManagementEntry

_PROVIDER_MAP: Dict[str, BaseProvider] = {
    "gpt": OpenAIProvider(),
    "o1": OpenAIProvider(),
    "claude": AnthropicProvider(),
    "nexus": NexusProvider(),
}


def _get_provider(model: str) -> BaseProvider:
    for prefix, provider in _PROVIDER_MAP.items():
        if model.startswith(prefix):
            return provider
    raise ValueError(f"No provider found for model: {model!r}")


def completion(
    model: str,
    messages: List[Dict[str, Any]],
    context_management: Optional[List[ContextManagementEntry]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    provider = _get_provider(model)

    # Each provider decides what to do with context_management
    messages, context_management = provider.transform_request(messages, context_management)

    supported = set(provider.get_supported_openai_params(model))
    if context_management and "context_management" in supported:
        kwargs["context_management"] = context_management

    non_default_params = {k: v for k, v in kwargs.items() if k in supported}
    optional_params: Dict[str, Any] = {}
    translated = provider.map_openai_params(non_default_params, optional_params, model)

    return {
        "model": model,
        "messages": messages,
        "provider": type(provider).__name__,
        **translated,
    }
