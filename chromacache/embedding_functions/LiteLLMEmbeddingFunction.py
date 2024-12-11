from abc import abstractmethod
import os
import time
from dotenv import load_dotenv
from litellm import embedding
from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

load_dotenv()


class LiteLLMEmbeddingFunction(AbstractEmbeddingFunction):
    """Base class for all embedding function dervied from litellm"""

    model_name: str
    dimensions: int | None
    input_type: str | None
    max_requests_per_minute: int | None

    def __init__(
        self,
        model_name: str,
        dimensions: int | None = None,
        input_type: str | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        AbstractEmbeddingFunction.__init__(self, model_name=model_name)
        self.dimensions = dimensions
        self.input_type = input_type
        self.sleep_time = (
            60 / max_requests_per_minute if max_requests_per_minute is not None else 0
        )
        self.check_api_key()

    @property
    @abstractmethod
    def api_key_name(self) -> str | list[str]:
        """the name of the environment variable containing the api key"""

    @property
    @abstractmethod
    def litellm_provider_prefix(self) -> str:
        """the prefix to use to specify provider in litellm"""

    @property
    def collection_name(self) -> str:
        return "_".join(
            (self.litellm_provider_prefix, f"dim-{self.dimensions}", self.model_name)
        )

    def check_api_key(self):
        """Ensure api key is set"""
        if not self.api_key_name:
            return
        self.api_key = os.environ.get(self.api_key_name, None)
        if self.api_key is None:
            raise ValueError(
                f"Please make sure {self.api_key_name} is setup as an environment variable"
            )

    def __call__(self, sentences: Documents) -> Embeddings:
        """Encodes the documents

        Args:
            documents (Documents): List of documents

        Returns:
            Embeddings: the encoded sentences
        """
        embeddings = self.encode_documents(sentences)
        if self.sleep_time:
            time.sleep(self.sleep_time)
        return embeddings

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
            input_type=self.input_type,
            dimensions=self.dimensions,
        )

        return [resp["embedding"] for resp in response.data]  # type: ignore --> missing typing for response.data