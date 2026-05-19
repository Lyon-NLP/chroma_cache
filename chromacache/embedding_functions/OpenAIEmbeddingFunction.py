import os

import openai
from dotenv import load_dotenv

from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

load_dotenv()


class OpenAIEmbeddingFunction(AbstractEmbeddingFunction):
    """Embedding function for OpenAI (and OpenAI-compatible) endpoints"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
        base_url: str | None = None,
        api_key_env_var: str = "OPENAI_API_KEY",
    ):
        AbstractEmbeddingFunction.__init__(
            self, model_name=model_name, max_requests_per_minute=max_requests_per_minute
        )
        if dimensions is not None and dimensions <= 0:
            raise ValueError("Argument 'dimensions' must be a positive integer.")
        self.dimensions = dimensions

        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Please make sure {api_key_env_var} is set as an environment variable"
            )
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @property
    def collection_name(self) -> str:
        return f"openai_dim-{self.dimensions}_{self.model_name}"

    def encode_documents(self, documents: Documents) -> Embeddings:
        """Get the embeddings for a list of documents.

        Args:
            documents (Documents): list of strings

        Returns:
            Embeddings: list of embeddings
        """
        documents = [d if d else " " for d in documents]
        kwargs = {}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(
            model=self.model_name,
            input=documents,
            **kwargs,
        )
        return [item.embedding for item in response.data]
