import os

from dotenv import load_dotenv

from .OpenAIEmbeddingFunction import OpenAIEmbeddingFunction

load_dotenv()


class ScalewayEmbeddingFunction(OpenAIEmbeddingFunction):
    """Embedding function for Scaleway AI endpoints (OpenAI-compatible API)"""

    def __init__(
        self,
        model_name: str = "bge-multilingual-gemma2",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        base_url = os.environ.get("SCW_ENDPOINT_EMBEDDING")
        if base_url is None:
            raise ValueError(
                "You must provide your Scaleway embedding endpoint as env variable 'SCW_ENDPOINT_EMBEDDING'"
            )
        OpenAIEmbeddingFunction.__init__(
            self,
            model_name=model_name,
            dimensions=dimensions,
            max_requests_per_minute=max_requests_per_minute,
            base_url=base_url,
            api_key_env_var="SCW_API_KEY",
        )

    @property
    def collection_name(self) -> str:
        return f"scaleway_dim-{self.dimensions}_{self.model_name}"
