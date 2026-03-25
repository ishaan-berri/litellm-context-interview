from abc import ABC, abstractmethod
from typing import Any, Dict, List


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
