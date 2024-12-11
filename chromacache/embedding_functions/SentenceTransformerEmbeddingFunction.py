from chromadb import Documents, Embeddings
import torch
from sentence_transformers import SentenceTransformer

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


class SentenceTransformerEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "dangvantuan/sentence-camembert-base",
        normalize_embeddings: bool = True,
    ):
        AbstractEmbeddingFunction.__init__(self, model_name=model_name)
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def litellm_provider_prefix(self):
        return None

    @property
    def api_key_name(self):
        return None

    def encode_documents(self, documents: Documents) -> Embeddings:
        embeddings = self.model.encode(
            documents, normalize_embeddings=self.normalize_embeddings
        )

        return embeddings.tolist()
