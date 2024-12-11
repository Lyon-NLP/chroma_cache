try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from abc import ABC, abstractmethod

from chromadb import Documents, EmbeddingFunction, Embeddings


class AbstractEmbeddingFunction(EmbeddingFunction, ABC):  # type: ignore --> missing typing in chromaDB
    """Base class for all embedding functions"""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Used as the collection name by chroma cache. Must lead to unique name per model"""
        return self.model_name

    def __call__(self, documents: Documents) -> Embeddings:
        """Encodes the documents

        Args:
            documents (Documents): List of documents

        Returns:
            Embeddings: the encoded sentences
        """
        return self.encode_documents(documents)

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
        raise NotImplementedError()
