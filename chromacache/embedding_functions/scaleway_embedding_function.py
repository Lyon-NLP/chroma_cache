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


class ScalewayEmbeddingFunction(AbstractEmbeddingFunction):
    """Embedding function for Scaleway AI endpoints"""

    def __init__(
        self,
        model_name: str = "bge-multilingual-gemma2",
        dimensions: int | None = None,
        max_requests_per_minute: int | None = None,
    ) -> None:
        AbstractEmbeddingFunction.__init__(
            self, model_name=model_name, max_requests_per_minute=max_requests_per_minute
        )
        if dimensions is not None and dimensions <= 0:
            raise ValueError("Argument 'dimensions' must be a positive integer.")
        self.dimensions = dimensions

        self.api_key = os.environ.get("SCW_API_KEY")
        if self.api_key is None:
            raise ValueError(
                "Please make sure SCW_API_KEY is setup as an environment variable"
            )
        self.endpoint = os.getenv("SCW_ENDPOINT_EMBEDDING")
        if self.endpoint is None:
            raise ValueError(
                "You must provide your Scaleway embedding endpoint as env variable 'SCW_ENDPOINT_EMBEDDING'"
            )

    @property
    def collection_name(self) -> str:
        return f"scaleway_dim-{self.dimensions}_{self.model_name}"

    def encode_documents(
        self,
        documents: Documents,
    ) -> Embeddings:
        """Get the embeddings for list of sentences

        Args:
            documents (Documents): list of sentences

        Raises:
            RuntimeError: If endpoint is not found (error 404)
            RuntimeError: If api doesn't answer with status 200
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"model": self.model_name, "input": documents}
        endpoint_response = _post_with_retry(
            self.endpoint, headers=headers, timeout=20, json=data
        )
        if endpoint_response.status_code == 404:
            raise RuntimeError(f"Endpoint {self.endpoint} not found.")
        if endpoint_response.status_code != 200:
            raise RuntimeError(endpoint_response.text)

        response = endpoint_response.json()

        if self.dimensions is not None:
            return [item["embedding"][: self.dimensions] for item in response["data"]]
        return [item["embedding"] for item in response["data"]]
