from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from litellm_mini.types import ContextManagementEntry


class BaseProvider(ABC):
    @abstractmethod
    def get_supported_openai_params(self, model: str) -> List[str]:
        ...

    @abstractmethod
    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        ...

    def transform_request(
        self,
        messages: List[Dict[str, Any]],
        context_management: Optional[List[ContextManagementEntry]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[ContextManagementEntry]]]:
        """
        Pre-process messages and context_management before map_openai_params.

        Default: return both unchanged.
        Providers that handle context_management client-side (e.g. Nexus)
        override this to trim messages and return None for context_management
        so the router knows not to forward it.
        """
        return messages, context_management
