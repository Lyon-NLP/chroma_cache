from typing import Literal

from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class CohereEmbeddingFunction(LiteLLMEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "embed-multilingual-light-v3.0",
        input_type: (
            Literal["search_document", "search_query", "classification", "clustering"]
            | None
        ) = None,
        dimensions: int | None = None,
        max_requests_per_minute: int = 2000,  # Prod, or 100 for trial
    ):
        LiteLLMEmbeddingFunction.__init__(
            model_name, dimensions, input_type, max_requests_per_minute
        )

    @property
    def api_key_name(self):
        return "COHERE_API_KEY"

    @property
    def litellm_provider_prefix(self):
        return "cohere"
