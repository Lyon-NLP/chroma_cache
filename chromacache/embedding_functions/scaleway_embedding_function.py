import os
import json
import requests

from chromadb import Documents, Embeddings
from .LiteLLMEmbeddingFunction import LiteLLMEmbeddingFunction


class ScalewayEmbeddingFunction(LiteLLMEmbeddingFunction):
    """Embedding function for OVH AI endpoints"""

    def __init__(
        self,
        model_name: str = "bge-multilingual-gemma2",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        LiteLLMEmbeddingFunction.__init__(
            self,
            model_name=model_name,
            dimensions=dimensions,
            max_requests_per_minute=max_requests_per_minute,
        )
        self.endpoint = os.getenv("SCW_ENDPOINT_EMBEDDING")
        if self.endpoint is None:
            raise ValueError(
                "You must provide your Scaleway embedding endpoint as env variable 'SCW_ENDPOINT_EMBEDDING'"
            )

    @property
    def api_key_name(self):
        return "SCW_API_KEY"

    @property
    def litellm_provider_prefix(self):
        return "scaleway"

    def encode_documents(
        self,
        documents: Documents,
    ) -> Embeddings:
        """Get the embeddings for list of sentences

        Args:
            documents (documents): list of sentences

        Raises:
            RuntimeError: If api doesn't answer with status 200 or 404
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": self.model_name, "input": documents}
        endpoint_response = requests.post(
            self.endpoint, headers=headers, json=data, timeout=20
        )
        if endpoint_response.status_code == 404:
            raise RuntimeError(f"Endpoint {self.endpoint} not found.")
        if endpoint_response.status_code != 200:
            raise RuntimeError(endpoint_response.text)

        response = json.loads(endpoint_response.text)

        if self.dimensions is not None:
            return [item["embedding"][: self.dimensions] for item in response["data"]]
        return [item["embedding"] for item in response["data"]]
