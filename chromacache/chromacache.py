try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
except Exception:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    import chromadb
    from chromadb import EmbeddingFunction
import tiktoken
from tqdm import tqdm


class ChromaCache:
    """Handles the mechanics of producing and saving embeddings in chromaDB
    It needs an embedding function, as described in the chromaDB's docs : https://docs.trychroma.com/embeddings
    This embedding function specifies the way embeddings are obtained, from a model or api
    """

    def __init__(
        self,
        embedding_function: EmbeddingFunction = None,
        batch_size: int = 32,
        save_embbedings: bool = True,
        path_to_chromadb: str = "./ChromaDB",
        max_token_length: int = 8191,
    ):
        self.batch_size = batch_size
        self.save_embbeddings = save_embbedings
        if embedding_function is None:
            raise ValueError(
                f"You must provide an embedding function. Embedding functions available are {chromadb.utils.embedding_functions.get_builtins()}. For more information, please visit : https://docs.trychroma.com/embeddings"
            )
        else:
            self.embedding_function = embedding_function

        # setup the chromaDB collection
        self.client = chromadb.PersistentClient(path=path_to_chromadb)
        collection_name = embedding_function.collection_name.replace("/", "-")[:63]
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self.max_token_length = max_token_length
        # Use tiktoken to compute token length
        # As we may not know the exact tokenizer used for the model, we generically use the one of adav2
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def list_embedding_functions():
        # TODO : function to list the available embedding functions
        raise NotImplementedError

    def encode(self, sentences: Documents) -> Embeddings:
        """Encodes the provided sentences et gets their embeddings
        using the EmbeddingFunction that has been set.
        It works like so :
        - If the string's embedding is already available in chroma, it gets it from there.
        - Otherwise, it uses the EmbeddingFunction that has been set up to get the embeddings
            while storing it in chroma for future reuse.

        Args:
            sentences (Documents): the list of strings that must be encoded

        Returns:
            Embeddings: the list of embeddings corresponding the the list of strings
        """
        # if sentences is a string, change to List[str]
        if isinstance(sentences, str):
            sentences = [sentences]

        # remove duplicate sentences, because chromaDB throws error if we get the same sentence
        # multiple times in one call
        unique_sentences = list(set(sentences))
        # use a dict to store a mapping of {sentence: embedding}
        # we have to do this because collection.get() returns embeddings in a random order...
        sent_emb_mapping = {}

        for i in tqdm(range(0, len(unique_sentences), self.batch_size)):
            # create batch and truncate
            batch_sentences = unique_sentences[i : i + self.batch_size]
            batch_sentences = self.truncate_documents(batch_sentences)
            # check if we have the embedding in chroma
            sentences_in_chroma = self.collection.get(
                ids=batch_sentences, include=["documents", "embeddings"]
            )

            # if we already have all the sentences in chroma, add those embeddings to the mapping dict
            if len(batch_sentences) == len(sentences_in_chroma["documents"]):
                sent_emb_mapping = sent_emb_mapping | dict(
                    zip(sentences_in_chroma["ids"], sentences_in_chroma["embeddings"])
                )

            # if we don't have all the sentences in chroma...
            else:
                missing_sentences = [
                    s for s in batch_sentences if s not in sentences_in_chroma["ids"]
                ]
                if self.save_embbeddings:
                    # We use the sentence as its own id in the database : not very clean, simplifies retrieving the sentence later
                    self.collection.upsert(
                        documents=missing_sentences, ids=missing_sentences
                    )
                    embs = self.collection.get(
                        ids=batch_sentences, include=["embeddings"]
                    )
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(embs["ids"], embs["embeddings"])
                    )
                else:
                    # first add what we have in chroma
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(
                            sentences_in_chroma["ids"],
                            sentences_in_chroma["embeddings"],
                        )
                    )
                    # then add what we obtain from encoding function
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(
                            missing_sentences,
                            self.embedding_function(missing_sentences),
                        )
                    )
        # return embeddings in correct order
        return [sent_emb_mapping[s] for s in sentences]

    def truncate_documents(self, sentences: Documents) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            sentences (Documents): a list a sentences (documents)

        Returns:
            Documents: the truncated documents
        """
        return [
            self.tokenizer.decode(self.tokenizer.encode(s)[: self.max_token_length])
            for s in sentences
        ]
