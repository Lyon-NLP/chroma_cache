import os
from abc import abstractmethod

from dotenv import load_dotenv
from litellm import embedding

from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

load_dotenv()


class LiteLLMEmbeddingFunction(AbstractEmbeddingFunction):
    """Base class for all embedding function derived from litellm"""

    model_name: str
    dimensions: int | None

    def __init__(
        self,
        model_name: str,
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        AbstractEmbeddingFunction.__init__(
            self, model_name=model_name, max_requests_per_minute=max_requests_per_minute
        )
        if dimensions is not None and dimensions <= 0:
            raise ValueError("Argument 'dimensions' must be a positive integer.")
        self.dimensions = dimensions
        self.check_api_key()

    @property
    @abstractmethod
    def api_key_name(self) -> str | list[str]:
        """The name(s) of the environment variable(s) containing the api key.

        Return a single string for one variable, or a list of strings when
        multiple variables are required (e.g. Azure needs API key, base, and
        version).
        """

    @property
    @abstractmethod
    def litellm_provider_prefix(self) -> str:
        """The prefix to use to specify provider in litellm"""

    @property
    def collection_name(self) -> str:
        return "_".join(
            (self.litellm_provider_prefix, f"dim-{self.dimensions}", self.model_name)
        )

    def check_api_key(self) -> None:
        """Ensure api key is set"""
        key_names = self.api_key_name
        if not key_names:
            return
        if isinstance(key_names, str):
            key_names = [key_names]
        missing = [k for k in key_names if not os.environ.get(k)]
        if missing:
            raise ValueError(
                f"Please make sure {', '.join(missing)} "
                f"{'are' if len(missing) > 1 else 'is'} set as an environment variable"
            )
        self.api_key: str = os.environ.get(key_names[0])

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
            dimensions=self.dimensions,
        )

        return [resp["embedding"] for resp in response.data]  # type: ignore --> missing typing for response.data
