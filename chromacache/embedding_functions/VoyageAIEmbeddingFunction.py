import os
import time

import voyageai as vai
from chromadb import Documents, Embeddings
from dotenv import load_dotenv

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

# load the API key from .env
load_dotenv()


class VoyageAIEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "voyage-code-2",
        max_token_length: int = 4000,
    ):
        super().__init__(max_token_length)

        self._model_name = model_name

        api_key = os.environ.get("VOYAGE_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Please make sure 'VOYAGE_API_KEY' is setup as an environment variable"
            )
        vai.api_key = api_key

        self.client = vai.Client()

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, documents: Documents) -> Embeddings:
        time.sleep(0.1)  # avoid api throttling
        return self.client.embed(
            documents, model=self._model_name, input_type=None, truncation=True
        ).embeddings
