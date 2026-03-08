from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache
from data.loader import load_dataset
from vectordb.vector_search import VectorSearch


app = FastAPI()

embedder = Embedder()

cache = SemanticCache()

documents = load_dataset()

doc_embeddings = embedder.embed(documents[:2000])

cluster_model = FuzzyCluster()

cluster_model.fit(doc_embeddings)

vector_search = VectorSearch(doc_embeddings, documents[:2000])


class QueryRequest(BaseModel):

    query: str


@app.post("/query")

def query(req: QueryRequest):

    query = req.query

    embedding = embedder.embed([query])[0]

    hit, entry, score = cache.lookup(embedding)

    if hit:

        return {

            "query": query,

            "cache_hit": True,

            "matched_query": entry["query"],

            "similarity_score": float(score),

            "result": entry["result"],

            "dominant_cluster": entry["cluster"]

        }

    cluster = cluster_model.dominant_cluster(embedding)

    result = vector_search.search(embedding)

    cache.store(query, embedding, result, cluster)

    return {

        "query": query,

        "cache_hit": False,

        "result": result,

        "dominant_cluster": cluster

    }


@app.get("/cache/stats")

def cache_stats():

    return cache.stats()


@app.delete("/cache")

def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}