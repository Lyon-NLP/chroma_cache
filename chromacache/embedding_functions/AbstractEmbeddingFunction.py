try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import time
from abc import ABC, abstractmethod

from chromadb import Documents, EmbeddingFunction, Embeddings


class AbstractEmbeddingFunction(EmbeddingFunction, ABC):  # type: ignore --> missing typing in chromaDB
    """Base class for all embedding functions"""

    def __init__(
        self,
        model_name: str,
        max_requests_per_minute: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_requests_per_minute = max_requests_per_minute

    @property
    def sleep_time(self) -> float:
        """Seconds to sleep between requests to stay within the rate limit."""
        return (
            60 / self.max_requests_per_minute
            if self.max_requests_per_minute is not None
            else 0
        )

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Used as the collection name by chroma cache. Must lead to unique name per model"""

    def __call__(self, documents: Documents) -> Embeddings:
        """Encodes the documents and applies rate-limit sleep if configured.

        The sleep accounts for the time already spent encoding, so only the
        remaining portion of the inter-request interval is waited.

        Args:
            documents (Documents): List of documents

        Returns:
            Embeddings: the encoded sentences
        """
        start = time.monotonic()
        embeddings = self.encode_documents(documents)
        if self.sleep_time:
            elapsed = time.monotonic() - start
            remaining = self.sleep_time - elapsed
            if remaining > 0:
                time.sleep(remaining)
        return embeddings

    @abstractmethod
    def encode_documents(self, documents: Documents) -> Embeddings:
        """Needs to be implemented by the child class. Takes a list of strings
        and returns the corresponding embedding

        Args:
            documents (Documents): list of documents (strings)

        Raises:
            NotImplementedError: Needs to be implements by child class

        Returns:
            Embeddings: list of embeddings
        """
        pass
