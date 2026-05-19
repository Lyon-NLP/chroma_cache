from .OpenAIEmbeddingFunction import OpenAIEmbeddingFunction


class OVHAIEmbeddingFunction(OpenAIEmbeddingFunction):
    """Embedding function for OVH AI endpoints (OpenAI-compatible API)"""

    def __init__(
        self,
        model_name: str = "multilingual-e5-base",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        OpenAIEmbeddingFunction.__init__(
            self,
            model_name=model_name,
            dimensions=dimensions,
            max_requests_per_minute=max_requests_per_minute,
            base_url=f"https://{model_name}.endpoints.kepler.ai.cloud.ovh.net/api",
            api_key_env_var="OVH_AI_ENDPOINTS_TOKEN",
        )

    @property
    def collection_name(self) -> str:
        return f"ovh_dim-{self.dimensions}_{self.model_name}"
