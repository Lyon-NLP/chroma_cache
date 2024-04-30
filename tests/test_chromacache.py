from __future__ import annotations

import logging
from typing import Union

import pytest

from chromacache import ChromaCache, SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO)

@pytest.mark.parametrize(
    "model_name",
    [
        "paraphrase-MiniLM-L6-v2",
        # "intfloat/multilingual-e5-small",
    ],
)
def test_encode(model_name: str):
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    chroma_cache = ChromaCache(embedding_function=embedding_function)
    embeddings = chroma_cache.encode("Hello, how are you?")

    assert isinstance(embeddings, Union[list, tuple])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
