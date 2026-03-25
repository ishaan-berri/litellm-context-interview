"""
Router: maps model names to providers, applies context_management.

Providers that list "context_management" in their supported params
(OpenAI, Anthropic) handle it natively — the param is forwarded.

Providers that do NOT list it (Nexus) have no API-side support.
For these, LiteLLM trims the messages client-side using the token trimmer
and does NOT forward context_management to the provider.

Your task: fill in the TODO below.
"""
from typing import Any, Dict, List, Optional

from litellm_mini.base_provider import BaseProvider
from litellm_mini.providers.anthropic import AnthropicProvider
from litellm_mini.providers.nexus import NexusProvider
from litellm_mini.providers.openai import OpenAIProvider
from litellm_mini.token_trimmer import get_compact_threshold, trim_messages
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
    supported = set(provider.get_supported_openai_params(model))

    if context_management:
        if "context_management" in supported:
            # Provider handles it natively — forward as a regular param
            kwargs["context_management"] = context_management
        else:
            # TODO: provider has no context_management support.
            # Use trim_messages to reduce the message list before dispatch.
            # Don't forward context_management to the provider.
            pass

    non_default_params = {k: v for k, v in kwargs.items() if k in supported}

    optional_params: Dict[str, Any] = {}
    translated = provider.map_openai_params(non_default_params, optional_params, model)

    return {
        "model": model,
        "messages": messages,
        "provider": type(provider).__name__,
        **translated,
    }
