from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class MistralAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    """Embedding function from mistral"""

    def __init__(
        self,
        model_name: str = "mistral-embed",
        dimensions: int | None = None,
        max_requests_per_minute: int = 300,
    ) -> None:
        LiteLLMEmbeddingFunction.__init__(
            self, model_name, dimensions, max_requests_per_minute
        )

    @property
    def api_key_name(self):
        return "MISTRAL_API_KEY"

    @property
    def litellm_provider_prefix(self):
        return "mistral"
