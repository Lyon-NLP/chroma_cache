from typing import Literal

from litellm import embedding

from chromadb import Documents, Embeddings

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

    @property
    def collection_name(self) -> str:
        # input_type is intentionally excluded: litellm doesn't forward it to
        # the API yet (see commented-out line in encode_documents), so all
        # input_type values produce identical embeddings and should share one
        # cache collection.  Re-add it here once litellm supports the param.
        return "_".join(
            [
                self.litellm_provider_prefix,
                f"dim-{self.dimensions}",
                self.model_name,
            ]
        )

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
