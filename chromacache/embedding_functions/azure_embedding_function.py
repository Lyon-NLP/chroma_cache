import os

from litellm import embedding
from chromadb import Documents, Embeddings
from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class AzureEmbeddingFunction(LiteLLMEmbeddingFunction):
    """Embedding function for Azure"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        LiteLLMEmbeddingFunction.__init__(
            self, model_name, dimensions, max_requests_per_minute
        )

    @property
    def api_key_name(self) -> list[str]:
        return ["AZURE_API_BASE", "AZURE_API_KEY", "AZURE_API_VERSION"]

    @property
    def litellm_provider_prefix(self) -> str:
        return "azure"

    def check_api_key(self):
        """Ensure api key is set"""
        self.azure_api_base = os.getenv(
            "AZURE_API_BASE"
        )  # e.g "https://<endpoint-url>.openai.azure.com"
        self.azure_api_key = os.getenv("AZURE_API_KEY")  # api key
        self.azure_api_version = os.getenv(
            "AZURE_API_VERSION"
        )  # api version, e.g. "2023-05-15"

        if not all([self.azure_api_base, self.azure_api_key, self.azure_api_version]):
            raise ValueError(
                f"Missing environment variable. Make sure {self.api_key_name} are set !"
            )

    def encode_documents(
        self,
        documents: Documents,
    ) -> Embeddings:
        """Get the embeddings for list of sentences

        Args:
            sentences (list[str]): list of sentences

        Raises:
            Exception: If api doesn't answer with status 200
        """
        response = embedding(
            model=f"{self.litellm_provider_prefix}/{self.model_name}",
            input=documents,
            api_key=self.azure_api_key,
            api_base=self.azure_api_base,
            api_version=self.azure_api_version,
            dimensions=self.dimensions,
        )

        return [resp["embedding"] for resp in response.data]  # type: ignore --> missing typing for response.data
