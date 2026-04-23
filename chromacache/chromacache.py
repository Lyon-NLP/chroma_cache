"""ChromaCache: caching layer for embedding functions backed by ChromaDB."""

try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
except ImportError:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
import hashlib
import re

import tiktoken
from tqdm import tqdm


class ChromaCache:
    """Caches embeddings in ChromaDB to avoid redundant API/model calls.

    Wraps any embedding function (see chromaDB docs) and transparently stores
    and retrieves embeddings so each unique text is only encoded once.
    """

    def __init__(
        self,
        embedding_function: EmbeddingFunction = None,
        batch_size: int = 32,
        save_embeddings: bool = True,
        path_to_chromadb: str = "./ChromaDB",
        max_token_length: int = 8191,
        tokenizer=None,
        verbose: bool = True,
    ):
        self.batch_size = batch_size
        self.save_embeddings = save_embeddings
        self.verbose = verbose
        if embedding_function is None:
            raise ValueError(
                "You must provide an embedding function. "
                "Call ChromaCache.list_embedding_functions() to see available options."
            )
        self.embedding_function = embedding_function

        # setup the chromaDB collection
        self.client = chromadb.PersistentClient(path=path_to_chromadb)
        collection_name = self._make_collection_name(embedding_function.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self.max_token_length = max_token_length
        # Use tiktoken (cl100k_base) by default. Note: this tokenizer matches OpenAI
        # models — for other providers (Cohere, SentenceTransformers, OVH, Scaleway…)
        # the token counts will differ, so pass a model-appropriate tokenizer if
        # precise truncation matters. A custom tokenizer must expose
        # encode(str) -> list[int] and decode(list[int]) -> str.
        self.tokenizer = (
            tokenizer if tokenizer is not None else tiktoken.get_encoding("cl100k_base")
        )

    @staticmethod
    def _make_collection_name(raw: str) -> str:
        """Return a ChromaDB-safe collection name (3-63 chars, unique).

        ChromaDB requires names to match ``[a-zA-Z0-9._-]+``, start and end
        with an alphanumeric character, contain no consecutive periods, and be
        between 3 and 63 characters long.  Any character that falls outside
        these rules is replaced with a hyphen, leading/trailing non-alphanumeric
        characters are stripped, and the result is padded to at least 3 chars.
        If the sanitised name exceeds 63 characters the first 46 characters are
        kept and a 16-character SHA-256 suffix is appended (joined by '_'),
        keeping the total at exactly 63 characters while preserving uniqueness.
        """
        # Replace every disallowed character with a hyphen
        sanitised = re.sub(r"[^a-zA-Z0-9._-]", "-", raw)
        # ChromaDB disallows consecutive periods
        sanitised = re.sub(r"\.{2,}", ".", sanitised)
        # Must start and end with an alphanumeric character
        sanitised = re.sub(r"^[^a-zA-Z0-9]+", "", sanitised)
        sanitised = re.sub(r"[^a-zA-Z0-9]+$", "", sanitised)
        # Enforce minimum length of 3
        if len(sanitised) < 3:
            sanitised = (sanitised + "000")[:3]
        if len(sanitised) <= 63:
            return sanitised
        name_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        truncated = sanitised[:46].rstrip("-_.")
        return f"{truncated}_{name_hash}"

    @staticmethod
    def _sentence_id(sentence: str) -> str:
        """Return a stable, ChromaDB-safe ID for a sentence."""
        return hashlib.sha256(sentence.encode()).hexdigest()

    @staticmethod
    def list_embedding_functions() -> list[str]:
        """Return the names of the available embedding function classes."""
        import chromacache.embedding_functions as ef
        from chromacache.embedding_functions import AbstractEmbeddingFunction

        return sorted(
            name
            for name in ef.__all__
            if isinstance(getattr(ef, name, None), type)
            and issubclass(getattr(ef, name), AbstractEmbeddingFunction)
            and getattr(ef, name) is not AbstractEmbeddingFunction
        )

    def encode(self, sentences: Documents) -> Embeddings:
        """Encode sentences, using the ChromaDB cache when possible.

        For each sentence:
        - if its embedding is already cached in ChromaDB, it is retrieved from there;
        - otherwise the embedding function is called and the result is optionally stored.

        Args:
            sentences (Documents): the list of strings to encode

        Returns:
            Embeddings: embeddings in the same order as the input
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # Truncate first so cache keys are always based on the stored (truncated) text
        sentences = self.truncate_documents(sentences)
        # deduplicate while preserving order
        unique_sentences = list(dict.fromkeys(sentences))
        # {sentence text: embedding} mapping built up across batches
        sent_emb_mapping: dict = {}

        for i in tqdm(
            range(0, len(unique_sentences), self.batch_size), disable=not self.verbose
        ):
            batch_sentences = unique_sentences[i : i + self.batch_size]
            batch_ids = [self._sentence_id(s) for s in batch_sentences]

            cached = self.collection.get(
                ids=batch_ids, include=["documents", "embeddings"]
            )

            # populate mapping from cache hits (document field holds original text)
            for doc, emb in zip(cached["documents"], cached["embeddings"]):
                sent_emb_mapping[doc] = emb

            if len(batch_sentences) == len(cached["documents"]):
                continue  # all cached

            cached_docs = set(cached["documents"])
            missing_sentences = [s for s in batch_sentences if s not in cached_docs]
            missing_ids = [self._sentence_id(s) for s in missing_sentences]

            if self.save_embeddings:
                # upsert lets ChromaDB call the embedding function internally
                self.collection.upsert(documents=missing_sentences, ids=missing_ids)
                embs = self.collection.get(
                    ids=missing_ids, include=["documents", "embeddings"]
                )
                for doc, emb in zip(embs["documents"], embs["embeddings"]):
                    sent_emb_mapping[doc] = emb
            else:
                for s, emb in zip(
                    missing_sentences, self.embedding_function(missing_sentences)
                ):
                    sent_emb_mapping[s] = emb

        return [sent_emb_mapping[s] for s in sentences]

    def truncate_documents(self, sentences: Documents) -> Documents:
        """Truncate sentences to fit within the model's context window.

        Args:
            sentences (Documents): list of sentences to truncate

        Returns:
            Documents: the truncated documents
        """
        return ChromaCache._truncate_documents(
            self.tokenizer, sentences, self.max_token_length
        )

    @staticmethod
    def _truncate_documents(
        tokenizer, sentences: Documents, max_token_length: int
    ) -> Documents:
        """Truncate sentences using the provided tokenizer.

        Args:
            tokenizer: object with encode(str) -> list[int] and decode(list[int]) -> str
            sentences (Documents): list of sentences to truncate
            max_token_length (int): maximum number of tokens to keep

        Returns:
            Documents: the truncated documents
        """
        return [
            tokenizer.decode(tokenizer.encode(s)[:max_token_length]) for s in sentences
        ]
