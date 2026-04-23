import warnings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction
from .azure_embedding_function import AzureEmbeddingFunction
from .CohereEmbeddingFunction import CohereEmbeddingFunction
from .MistralAIEmbeddingFunction import MistralAIEmbeddingFunction
from .OpenAIEmbeddingFunction import OpenAIEmbeddingFunction
from .ovh_embedding_function import OVHAIEmbeddingFunction
from .scaleway_embedding_function import ScalewayEmbeddingFunction
from .VoyageAIEmbeddingFunction import VoyageAIEmbeddingFunction

__all__ = [
    "AbstractEmbeddingFunction",
    "CohereEmbeddingFunction",
    "MistralAIEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "VoyageAIEmbeddingFunction",
    "AzureEmbeddingFunction",
    "OVHAIEmbeddingFunction",
    "ScalewayEmbeddingFunction",
]

try:
    from .SentenceTransformerEmbeddingFunction import (
        SentenceTransformerEmbeddingFunction as SentenceTransformerEmbeddingFunction,
    )

    __all__.append("SentenceTransformerEmbeddingFunction")
except ModuleNotFoundError:
    warnings.warn(
        "'torch' and/or 'sentence-transformers' modules not found. "
        "SentenceTransformerEmbeddingFunction won't be available. "
        "If needed: pip install chromacache[st]",
        ImportWarning,
        stacklevel=2,
    )
