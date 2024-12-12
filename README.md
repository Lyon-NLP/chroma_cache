# ChromaCache

## Motivation
This module levereges chromaDB to cache embeddings for reusability. 
It helps storing and fetching embeddings of chromaDB.

In a nutshell :
- it creates a vector store with the model name / provider as a collection name
- it encodes any sentence using the specified embedding function using the ``encode()`` method
- among the sentences provided in the ``encode`` method, it calls the model (or the api) to embed those for which the embedding are not already available in chromaDB

## Installation

Here are the installation steps:
- 1) If you haven't already, clone this repository.
- 2) Activate your python environment (or shell)
- 3) when your are in this repository, run ``pip install .`` to install this repo as a package
- 4) [Optionnal] To make the package lighter, sentence-transformers dependencies are optionnal. If you are planning to use models from HuggingFace 🤗 through the sentence-transfomer package, you may use ``pip install ".[st]"`` instead.

***TODO*** : Add instruction for installation as pypi package

## Usage 

Fast & simple usage :

```py
from chromacache import ChromaCache
from chromacache.embedding_functions import OpenAIEmbeddingFunction # or any embedding function available

cc = ChromaCache(OpenAIEmbeddingFunction())
embeddings = cc.encode(["my sentence", "my other sentence"])
```

Usage when using huggingface's ``datasets`` package.

```py
import datasets
from chromacache import ChromaCache
from chromacache.embedding_functions import AzureEmbeddingFunction  # or any embedding function available

emb_function = AzureEmbeddingFunction(
    model_name="text-embedding-3-large",
    dimensions=768,
    max_requests_per_minute=300,
)
cc = ChromaCache(
    emb_func,
    batch_size=16,
    path_to_chromadb="path/to/my/chromadb/folder",
    max_token_length=8191 # adapt to the model you use
    )
# let's assume this return a 'Dataset' object, with a 'text' column we want to embed
mydataset = datasets.load_dataset("PathToMyDataset/OnHuggingFace")
mydataset = mydataset.add_column(
    "embeddings", cc.encode(mydataset["text"])
    )
```

## Extra arguments

The ``ChromaCache`` supports extra arguments :
- ***batch_size***: int = 32, the batch size at which sentences are processed. If the model's provider API raises an error due to the size of the request being exceeded, it might be a good idea to decrease this
- ***save_embbedings***: bool = True, whether or not the embeddings should be saved
- ***path_to_chromadb***: str = "./Chroma", where the chromadb should be stored
- ***max_token_length***: int = 8191, texts longer than this amount of tokens will be truncated to avoid API Errors.

Moreover, all [capabilities of the chromaDB collections](https://docs.trychroma.com/reference/py-collection) can be leveraged directly using the ``collection`` attribute of the ChromaCache. Hence, you can query, delete, ... any collection.
For example, to query the collection for the 5 documents:
```py
cc = ChromaCache(VoyageAIEmbeddingFunction("voyage-code-2"))
relevant_documents = cc.collection.query(
    query_texts=["my query1", "thus spake zarathustra", ...],
    n_results=5,
)
```