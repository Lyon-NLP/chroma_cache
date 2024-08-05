# ChromaCache

## Motivation
This module levereges chromaDB to cache embeddings for reusability. 
It helps storing and fetching embeddings of chromaDB.

In a nutshell :
- it creates a vector store with the model name as a collection name
- it encodes any sentence using the specified embedding function using the ``encode()`` method
- among the sentences provided in the ``encode`` method, it calls the model (or the api) to embed those for which the embedding are not already available in chromaDB

## Installation

Here are the installation steps:
- 1) If you haven't already, clone this repository.
- 2) Activate your python environment (or shell)
- 3) when your are in this repository, run ``pip install .`` to install this repo as a package

⚠️ if you run into an error due to faiseq, this is du to incompatibility between fairseq, win11 and python > 3.9. Either downgrade your python version, or install this fix in your environment : 
```bash
pip install fairseq git+https://github.com/liyaodev/fairseq.git 
```

***TODO*** : Add instruction for installation as pypi package

## Usage 

```py
from chromacache import ChromaCache
from chromacache.embedding_functions import OpenAIEmbeddingFunction

MODEL_NAME = "text-embedding-3-small" # or any embedding model name
emb_function = OpenAIEmbeddingFunction() # or any embedding function available
cc = ChromaCache(OpenAIEmbeddingFunction(MODEL_NAME)) # creates a collection in chroma

embeddings = cc.encode(["my sentence", "my other sentence"])
```

## Extra features

The ``ChromaCache`` supports extra arguments :
- ***batch_size***: int = 32, the batch size at which sentences are processed. If the model's provider API raises an error due to the size of the request being exceeded, it might be a good idea to decrease this
- ***save_embbedings***: bool = True, whether or not the embeddings should be saved
- ***path_to_chromadb***: str = "./Chroma", where the chromadb should be stored

All embedding functions also support the ``max_token_length`` argument. This can be used to crop each sentence to the max token size supported by the model's provider API

Example usage :
```py
emb_function = MistralAIEmbeddingFunction("mistral-embed", max_token_length=4000)
cc = ChromaCache(
    emb_func,
    batch_size=4,
    save_embedding=False,
    path_to_chromdb="./my_favorite_directory"
    )
```

Moreover, all capabilities of the chromaDB collections can be leveraged directly using the ``collection`` attribute of the ChromaCache.
For example, to query the collection for the 5 documents:
```py
cc = ChromaCache(VoyageAIEmbeddingFunction("voyage-code-2"))
relevant_documents = cc.collection.query(
    query_texts=["my query1", "thus spake zarathustra", ...],
    n_results=5,
)
```