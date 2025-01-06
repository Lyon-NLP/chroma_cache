from typing import Literal

from chromadb import Documents, Embeddings
from litellm import embedding

from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class VoyageAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "voyage-code-2",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
        input_type: Literal["query", "document"] | None = None,
    ):
        LiteLLMEmbeddingFunction.__init__(
            self, model_name, dimensions, max_requests_per_minute
        )
        self.input_type = input_type

    @property
    def api_key_name(self) -> str:
        return "VOYAGE_API_KEY"

    @property
    def litellm_provider_prefix(self) -> str:
        return "voyage"

    def encode_documents(self, documents: Documents) -> Embeddings:
        """Takes a list of strings and returns the corresponding embedding

        Args:
            documents (Documents): list of documents (strings)

        Returns:
            Embeddings: list of embeddings
        """
        # replace empty string to avoid errors with apis
        documents = [d if d else " " for d in documents]
        response = embedding(
            model=f"{self.litellm_provider_prefix}/{self.model_name}",
            input=documents,
            # input_type=self.input_type, # Kept here as a placeholder. Unfortunately litellm doesn't allow for passing this param yet.
            dimensions=self.dimensions,
        )

        return [resp["embedding"] for resp in response.data]  # type: ignore --> missing typing for response.data
