from __future__ import annotations

import logging
from typing import Union

import pytest

from chromacache import ChromaCache, SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO)


def _emb_eq(a, b) -> bool:
    """Compare two embeddings that may be numpy arrays or plain lists."""
    try:
        import numpy as np
        return np.array_equal(a, b)
    except ImportError:
        return list(a) == list(b)

MODEL_NAME = "paraphrase-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@pytest.fixture
def tmp_cache(tmp_path):
    """Return a ChromaCache backed by a temporary directory (no state pollution)."""
    ef = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    return ChromaCache(
        embedding_function=ef,
        path_to_chromadb=str(tmp_path / "chromadb"),
        verbose=False,
    )


@pytest.mark.parametrize(
    "model_name",
    [
        MODEL_NAME,
        # "intfloat/multilingual-e5-small",
    ],
)
def test_encode(model_name: str):
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
    chroma_cache = ChromaCache(
        embedding_function=embedding_function,
        verbose=False,
    )
    embeddings = chroma_cache.encode("Hello, how are you?")

    assert isinstance(embeddings, Union[list, tuple])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Output ordering
# ---------------------------------------------------------------------------


def test_encode_order_preserved(tmp_cache):
    """Embeddings must be returned in the same order as the input sentences."""
    sentences = [
        "The cat sat on the mat.",
        "A quick brown fox jumps over the lazy dog.",
        "Hello world.",
        "Machine learning is fascinating.",
    ]
    embeddings = tmp_cache.encode(sentences)

    assert len(embeddings) == len(sentences)

    # Encode each sentence individually and verify the batch result matches
    for i, sentence in enumerate(sentences):
        single = tmp_cache.encode([sentence])
        assert _emb_eq(embeddings[i], single[0]), (
            f"Embedding at position {i} does not match individual encoding of: {sentence!r}"
        )


def test_encode_order_preserved_reversed(tmp_cache):
    """Reversing the input must reverse the output, not produce the same order."""
    sentences = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    forward = tmp_cache.encode(sentences)
    backward = tmp_cache.encode(list(reversed(sentences)))

    assert _emb_eq(forward[0], backward[2])
    assert _emb_eq(forward[1], backward[1])
    assert _emb_eq(forward[2], backward[0])


# ---------------------------------------------------------------------------
# Cache retrieval (hit vs miss)
# ---------------------------------------------------------------------------


def test_embeddings_retrieved_from_cache(tmp_path):
    """Embeddings fetched from cache must be identical to freshly computed ones."""
    ef = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    db_path = str(tmp_path / "chromadb")

    sentences = ["Cache me if you can.", "Another sentence."]

    # First call: computes and stores
    cache1 = ChromaCache(embedding_function=ef, path_to_chromadb=db_path, verbose=False)
    first_embeddings = cache1.encode(sentences)

    # Second call with a new ChromaCache instance pointing to the same DB
    cache2 = ChromaCache(embedding_function=ef, path_to_chromadb=db_path, verbose=False)
    second_embeddings = cache2.encode(sentences)

    assert len(first_embeddings) == len(second_embeddings)
    for a, b in zip(first_embeddings, second_embeddings):
        assert _emb_eq(a, b)


def test_embedding_shape(tmp_cache):
    """Every returned embedding must be a list of floats with the expected dimension."""
    sentences = ["Shape test one.", "Shape test two.", "Shape test three."]
    embeddings = tmp_cache.encode(sentences)

    assert len(embeddings) == len(sentences)
    for emb in embeddings:
        assert len(emb) == EMBEDDING_DIM
        assert all(isinstance(v, (float, int)) or hasattr(v, "item") for v in emb)


# ---------------------------------------------------------------------------
# Duplicate inputs
# ---------------------------------------------------------------------------


def test_duplicate_sentences_return_identical_embeddings(tmp_cache):
    """Duplicate sentences in the input must produce identical embeddings."""
    sentence = "Duplicated input sentence."
    sentences = [sentence, "Other sentence.", sentence]

    embeddings = tmp_cache.encode(sentences)

    assert len(embeddings) == 3
    assert _emb_eq(embeddings[0], embeddings[2])


def test_all_duplicates(tmp_cache):
    """Input made entirely of the same sentence should return consistent embeddings."""
    sentence = "All the same."
    embeddings = tmp_cache.encode([sentence] * 5)

    assert len(embeddings) == 5
    assert all(_emb_eq(e, embeddings[0]) for e in embeddings)


# ---------------------------------------------------------------------------
# Single string input
# ---------------------------------------------------------------------------


def test_single_string_input(tmp_cache):
    """Passing a bare string (not a list) must be handled gracefully."""
    embeddings = tmp_cache.encode("Just a single string.")

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# save_embeddings=False path
# ---------------------------------------------------------------------------


def test_encode_without_saving(tmp_path):
    """With save_embeddings=False the cache should still return correct embeddings."""
    ef = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    cache = ChromaCache(
        embedding_function=ef,
        path_to_chromadb=str(tmp_path / "chromadb"),
        save_embeddings=False,
        verbose=False,
    )
    sentences = ["No save one.", "No save two."]
    embeddings = cache.encode(sentences)

    assert len(embeddings) == len(sentences)
    for emb in embeddings:
        assert len(emb) == EMBEDDING_DIM
