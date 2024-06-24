import os

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction as OAIEMbFunc
from dotenv import load_dotenv

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

# load the API key from .env
load_dotenv()


class OpenAIEmbeddingFunction(AbstractEmbeddingFunction, OAIEMbFunc):
    def __init__(
        self,
        model_name: str = "text-embedding-small",
        max_token_length: int = 8191,
    ):
        AbstractEmbeddingFunction.__init__(self, max_token_length=max_token_length)

        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Please make sure 'OPENAI_API_KEY' is setup as an environment variable"
            )
        OAIEMbFunc.__init__(self, api_key=api_key, model_name=model_name)

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, documents: Documents) -> Embeddings:
        # replace empty string by space to avoid API error
        documents = [i if i else " " for i in documents]
        return OAIEMbFunc.__call__(self, documents)
