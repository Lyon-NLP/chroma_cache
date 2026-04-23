from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


class SentenceTransformerEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "dangvantuan/sentence-camembert-base",
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
    ):
        """Initialize the SentenceTransformer embedding function.

        Args:
            model_name: HuggingFace model identifier.
            normalize_embeddings: Whether to L2-normalise the output vectors.
            trust_remote_code: Allow executing model-provided code from the
                HuggingFace Hub.  Only set to True for models you explicitly
                trust, as this runs arbitrary Python code from the repository.
        """
        AbstractEmbeddingFunction.__init__(self, model_name=model_name)
        self.normalize_embeddings = normalize_embeddings

        # Lazy imports so that missing torch/sentence-transformers only raises
        # at instantiation time, not at module import time.
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=trust_remote_code,
        )

    @property
    def collection_name(self):
        return "st_" + self.model_name

    def encode_documents(self, documents: Documents) -> Embeddings:
        embeddings = self.model.encode(
            documents, normalize_embeddings=self.normalize_embeddings
        )

        return embeddings.tolist()
