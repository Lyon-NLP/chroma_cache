import json
import requests

from chromadb import Documents, Embeddings
from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class OVHAIEmbeddingFunction(LiteLLMEmbeddingFunction):
    """Embedding function for OVH AI endpoints"""

    def __init__(
        self,
        model_name: str = "multilingual-e5-base",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        LiteLLMEmbeddingFunction.__init__(
            self,
            model_name=model_name,
            dimensions=dimensions,
            max_requests_per_minute=max_requests_per_minute,
        )
        self.endpoint = (
            f"https://{model_name}.endpoints.kepler.ai.cloud.ovh.net/api/batch_text2vec"
        )

    @property
    def api_key_name(self):
        return "OVH_AI_ENDPOINTS_TOKEN"

    @property
    def litellm_provider_prefix(self):
        return "ovh"

    def encode_documents(
        self,
        documents: Documents,
    ) -> Embeddings:
        """Get the embeddings for list of sentences

        Args:
            sentences (list[str]): list of sentences

        Raises:
            RuntimeError: If endpoint is not found (error 404)
            RuntimeError: If api doesn't answer with status 200 or 404
        """
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.endpoint, data=json.dumps(documents), headers=headers, timeout=30
        )
        if response.status_code == 200:
            embeddings = response.json()
            if self.dimensions is not None:
                return [emb[: self.dimensions] for emb in embeddings]
            return embeddings
        if response.status_code == 404:
            raise RuntimeError(f"Endpoint {self.endpoint} not found.")
        raise RuntimeError("Problem with API")
