from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class OpenAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    """Embedding function for OpenAI"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        input_type: str | None = None,
        max_requests_per_minute: int | None = None,
    ):
        LiteLLMEmbeddingFunction.__init__(
            model_name, dimensions, input_type, max_requests_per_minute
        )

    @property
    def api_key_name(self):
        return "OPENAI_API_KEY"

    @property
    def litellm_provider_prefix(self):
        return "openai"
