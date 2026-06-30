"""
ChromaDB RAG helper for the Synthesiser agent. Embeds each approved paper's
abstract with sentence-transformers and retrieves the top-k most relevant
abstracts per sub-question, rather than stuffing every abstract into the
prompt verbatim.

Uses an in-memory ephemeral client — swap chromadb.Client() for
chromadb.PersistentClient(path="./.chroma") if you want the index to survive
across runs.
"""
import chromadb
from chromadb.utils import embedding_functions

_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def build_collection(papers: list[dict], collection_name: str = "scholarmind"):
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, embedding_function=_EMBED_FN)

    ids, docs, metadatas = [], [], []
    for p in papers:
        if not p.get("abstract"):
            continue
        ids.append(p["paper_id"])
        docs.append(p["abstract"])
        metadatas.append(
            {"title": p["title"], "year": p.get("year") or 0, "stance": p.get("stance", 0)}
        )

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
    return collection


def retrieve(collection, query: str, k: int = 4) -> list[dict]:
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(k, count))
    return [
        {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
