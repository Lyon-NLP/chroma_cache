import pytest
import tiktoken

from chromacache import ChromaCache
from chromacache.embedding_functions import AbstractEmbeddingFunction


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Minimal concrete implementation used across tests
# ---------------------------------------------------------------------------


class DummyEmbeddingFunction(AbstractEmbeddingFunction):
    """Deterministic stub: embedding of a document is its character-code list padded to DIM."""

    DIM = 8

    def __init__(self):
        super().__init__(model_name="dummy-model")
        self.call_count = 0

    @property
    def collection_name(self) -> str:
        return "dummy-model"

    def encode_documents(self, documents):
        self.call_count += 1
        embeddings = []
        for doc in documents:
            raw = [float(ord(c)) for c in doc[: self.DIM]]
            padded = raw + [0.0] * (self.DIM - len(raw))
            embeddings.append(padded)
        return embeddings


@pytest.fixture
def dummy_ef():
    return DummyEmbeddingFunction()


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentences",
    [
        [
            "This is a very long sentence with more than 10 tokens and it should be truncated."
        ],
        [
            "This is a very long sentence with more than 10 tokens and it should be truncated.",
            "This is a second very long sentence with more than 10 tokens and it should be truncated.",
        ],
    ],
)
def test_truncate_documents(tokenizer, sentences):
    max_token_length = 10

    truncated_sentences = ChromaCache._truncate_documents(
        tokenizer, sentences, max_token_length
    )

    assert len(truncated_sentences) == len(sentences)

    tokenized_truncated_sentences = [tokenizer.encode(s) for s in truncated_sentences]
    for truncated_sentence in tokenized_truncated_sentences:
        assert len(truncated_sentence) <= max_token_length


def test_truncate_preserves_order(tokenizer):
    """Truncation must not reorder the sentences."""
    sentences = ["Alpha beta gamma delta epsilon.", "One two three.", "Short."]
    truncated = ChromaCache._truncate_documents(tokenizer, sentences, 3)

    assert len(truncated) == len(sentences)
    # Shorter originals should remain unchanged (≤ 3 tokens already)
    assert truncated[2] == "Short."


# ---------------------------------------------------------------------------
# AbstractEmbeddingFunction: __call__ and encode_documents
# ---------------------------------------------------------------------------


def _to_list(embedding) -> list:
    """Normalise an embedding to a plain Python list for comparison."""
    try:
        return embedding.tolist()
    except AttributeError:
        return list(embedding)


def test_call_returns_same_as_encode_documents(dummy_ef):
    """`__call__` must return the same embeddings as `encode_documents`."""
    docs = ["hello", "world"]
    via_call = [_to_list(e) for e in dummy_ef(docs)]
    via_encode = [_to_list(e) for e in dummy_ef.encode_documents(docs)]
    assert via_call == via_encode


def test_embedding_output_order(dummy_ef):
    """Embeddings must be returned in the same order as the input documents."""
    docs = ["abc", "xyz", "mno"]
    embeddings = dummy_ef(docs)

    assert len(embeddings) == len(docs)
    for doc, emb in zip(docs, embeddings):
        expected = _to_list(dummy_ef.encode_documents([doc])[0])
        assert _to_list(emb) == expected, f"Order mismatch for document {doc!r}"


def test_embedding_dim_consistent(dummy_ef):
    """All returned embeddings must have the same dimension."""
    docs = ["a", "longer document here", "x" * 20]
    embeddings = dummy_ef(docs)

    dims = [len(e) for e in embeddings]
    assert len(set(dims)) == 1, f"Inconsistent embedding dims: {dims}"


def test_single_document_embedding(dummy_ef):
    """A single-document input should return a list with one embedding."""
    embeddings = dummy_ef(["single"])
    assert len(embeddings) == 1
    assert hasattr(embeddings[0], "__len__")  # list or numpy array


def test_sleep_time_none(dummy_ef):
    """`sleep_time` should be 0 when no rate limit is configured."""
    assert dummy_ef.sleep_time == 0


def test_sleep_time_with_rate_limit():
    """sleep_time must equal 60 / max_requests_per_minute."""
    ef = DummyEmbeddingFunction()
    ef.max_requests_per_minute = 30
    assert ef.sleep_time == pytest.approx(2.0)


def test_collection_name_property(dummy_ef):
    """collection_name must return a non-empty string."""
    name = dummy_ef.collection_name
    assert isinstance(name, str) and len(name) > 0


def test_abstract_cannot_be_instantiated():
    """AbstractEmbeddingFunction cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AbstractEmbeddingFunction(model_name="x")  # type: ignore[abstract]
