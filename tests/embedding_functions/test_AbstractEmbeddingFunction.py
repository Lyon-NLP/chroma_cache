import pytest
import tiktoken

from chromacache import AbstractEmbeddingFunction


@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("cl100k_base")


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

    truncated_sentences = AbstractEmbeddingFunction._truncate_documents(
        tokenizer, sentences, max_token_length
    )

    assert len(truncated_sentences) == len(sentences)

    tokenized_truncated_sentences = [tokenizer.encode(s) for s in truncated_sentences]
    for truncated_sentence in tokenized_truncated_sentences:
        assert len(truncated_sentence) <= max_token_length
