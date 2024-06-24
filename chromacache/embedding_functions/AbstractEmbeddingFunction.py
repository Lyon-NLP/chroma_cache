from __future__ import annotations

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

from abc import ABC, abstractmethod
import tiktoken
from chromadb import EmbeddingFunction, Documents, Embeddings


class AbstractEmbeddingFunction(EmbeddingFunction, ABC):
    REGISTRY: dict[str, AbstractEmbeddingFunction] = {}

    def __init__(
        self,
        max_token_length: int = 4096,
    ):
        self.max_token_length = max_token_length
        # Use tiktoken to compute token length
        # As we may not know the exact tokenizer used for the model, we generically use the one of adav2
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    @abstractmethod
    def model_name(self):
        pass

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     # cls.register(cls)
    #     name = cls.__class__.__name__
    #     cls.REGISTRY[name] = cls

    @classmethod
    def register(cls, class_: AbstractEmbeddingFunction) -> None:
        name = class_.__class__.__name__
        cls.REGISTRY[name] = class_
    
    @classmethod
    def get(cls, name: str) -> AbstractEmbeddingFunction:
        return cls.REGISTRY[name]

    @classmethod
    def get_all_embedding_functions_names(cls) -> list[str]:
        return list(cls.REGISTRY.keys())
    
    @classmethod
    def get_all_embedding_functions(cls) -> list[AbstractEmbeddingFunction]:
        return list(cls.REGISTRY.values())

    def truncate_documents(self, sentences: Documents) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            sentences (Documents): a list a sentences (documents)

        Returns:
            Documents: the truncated documents
        """
        truncated_input = []
        for s in sentences:
            tokenized_string = self.tokenizer.encode(s)
            # if string too large, truncate, decode, and replace
            if len(tokenized_string) > self.max_token_length:
                tokenized_string = tokenized_string[: self.max_token_length]
                truncated_input.append(self.tokenizer.decode(tokenized_string))
            else:
                truncated_input.append(s)

        return truncated_input

    def __call__(self, input: Documents) -> Embeddings:
        """Wrapper that truncates the documents, encodes them

        Args:
            documents (Documents): List of documents

        Returns:
            Embeddings: the encoded sentences
        """
        truncated_input = self.truncate_documents(input)
        embeddings = self.encode_documents(truncated_input)

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

        raise NotImplementedError()
