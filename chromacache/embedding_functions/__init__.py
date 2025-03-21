from .AbstractEmbeddingFunction import AbstractEmbeddingFunction
from .CohereEmbeddingFunction import CohereEmbeddingFunction
from .MistralAIEmbeddingFunction import MistralAIEmbeddingFunction
from .OpenAIEmbeddingFunction import OpenAIEmbeddingFunction
from .VoyageAIEmbeddingFunction import VoyageAIEmbeddingFunction
from .azure_embedding_function import AzureEmbeddingFunction
from .ovh_embedding_function import OVHAIEmbeddingFunction
from .scaleway_embedding_function import ScalewayEmbeddingFunction

try:
    from .SentenceTransformerEmbeddingFunction import (
        SentenceTransformerEmbeddingFunction,
    )
except ModuleNotFoundError:
    print(
        "'torch' and/or 'sentence-transfomers' modules not found.\n",
        "SentenceTransformerEmbeddingFunction won't be available.\n",
        "If needed : pip install chromacache[st]",
    )
