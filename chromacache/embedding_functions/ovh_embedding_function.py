import json
import os
import time

import requests
from dotenv import load_dotenv

from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

load_dotenv()

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def _post_with_retry(
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int = 3,
    **kwargs,
) -> requests.Response:
    """POST with exponential-backoff retry on transient errors."""
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, timeout=timeout, **kwargs)
        if (
            response.status_code not in _RETRYABLE_STATUS_CODES
            or attempt == max_retries - 1
        ):
            return response
        time.sleep(2**attempt)
    return response  # unreachable


class OVHAIEmbeddingFunction(AbstractEmbeddingFunction):
    """Embedding function for OVH AI endpoints"""

    def __init__(
        self,
        model_name: str = "multilingual-e5-base",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        AbstractEmbeddingFunction.__init__(
            self, model_name=model_name, max_requests_per_minute=max_requests_per_minute
        )
        if dimensions is not None and dimensions <= 0:
            raise ValueError("Argument 'dimensions' must be a positive integer.")
        self.dimensions = dimensions

        self.api_key = os.environ.get("OVH_AI_ENDPOINTS_TOKEN")
        if self.api_key is None:
            raise ValueError(
                "Please make sure OVH_AI_ENDPOINTS_TOKEN is setup as an environment variable"
            )
        self.endpoint = (
            f"https://{model_name}.endpoints.kepler.ai.cloud.ovh.net/api/batch_text2vec"
        )

    @property
    def collection_name(self) -> str:
        return f"ovh_dim-{self.dimensions}_{self.model_name}"

    def encode_documents(
        self,
        documents: Documents,
    ) -> Embeddings:
        """Get the embeddings for list of sentences

        Args:
            documents (Documents): list of sentences

        Raises:
            RuntimeError: If endpoint is not found (error 404)
            RuntimeError: If api doesn't answer with status 200 or 404
        """
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = _post_with_retry(
            self.endpoint, headers=headers, timeout=30, data=json.dumps(documents)
        )
        if response.status_code == 200:
            embeddings = response.json()
            if self.dimensions is not None:
                return [emb[: self.dimensions] for emb in embeddings]
            return embeddings
        if response.status_code == 404:
            raise RuntimeError(f"Endpoint {self.endpoint} not found.")
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
