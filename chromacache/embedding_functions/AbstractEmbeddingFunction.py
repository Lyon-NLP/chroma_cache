try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from abc import ABC, abstractmethod

import tiktoken
from chromadb import Documents, EmbeddingFunction, Embeddings


class AbstractEmbeddingFunction(EmbeddingFunction, ABC):
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

    @staticmethod
    def _truncate_documents(
        tokenizer, sentences: Documents, max_token_length: int
    ) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            tokenizer : the tokenizer
            sentences (Documents): a list a sentences (documents)
            max_token_length (int): the maximum token length

        Returns:
            Documents: the truncated documents
        """
        truncated_input = []
        for s in sentences:
            tokenized_string = tokenizer.encode(s)
            # if string too large, truncate, decode, and replace
            if len(tokenized_string) > max_token_length:
                tokenized_string = tokenized_string[:max_token_length]
                truncated_input.append(tokenizer.decode(tokenized_string))
            else:
                truncated_input.append(s)

        return truncated_input

    def truncate_documents(self, sentences: Documents) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            sentences (Documents): a list a sentences (documents)

        Returns:
            Documents: the truncated documents
        """
        return self._truncate_documents(
            self.tokenizer, sentences, self.max_token_length
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Wrapper that truncates the documents, encodes them

        Args:
            input (Documents): List of documents

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
