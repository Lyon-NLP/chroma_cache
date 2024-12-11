from typing import Literal

from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class VoyageAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "voyage-code-2",
        max_requests_per_minute: int | None = None,
        input_type: Literal["query", "document"] | None = None,
        dimensions: int | None = None,
    ):
        LiteLLMEmbeddingFunction.__init__(
            model_name, dimensions, input_type, max_requests_per_minute
        )

    @property
    def litellm_provider_prefix(self) -> str:
        return "voyage"

    @property
    def api_key_name(self) -> str:
        return "VOYAGE_API_KEY"
