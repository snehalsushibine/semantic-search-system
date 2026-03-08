# Semantic Search System

This project implements a lightweight semantic search system built on the 20 Newsgroups dataset (~20,000 documents).

The system demonstrates how semantic embeddings, fuzzy clustering, and a custom semantic cache can be combined to build an efficient query system exposed through a FastAPI service.


## Features

- Semantic text embeddings using SentenceTransformers
- Fuzzy clustering of documents using Gaussian Mixture Models
- Vector similarity search over document embeddings
- Custom semantic cache built from scratch (no Redis or external caching system)
- FastAPI REST API for querying and cache management
- Cluster visualization for exploratory analysis


## Dataset

The system uses the 20 Newsgroups dataset, a collection of approximately 20,000 news posts across 20 categories.

The dataset is automatically downloaded using `scikit-learn`.


## System Architecture

User Query  
↓  
Embedding Model (SentenceTransformer)  
↓  
Semantic Cache Lookup  
↓  
Cache Hit → Return Cached Result  

Cache Miss →  
→ Semantic Vector Search  
→ Fuzzy Cluster Assignment  
→ Store Result in Cache  

↓

Return Response via FastAPI


## Project Structure
Tassignment
│
├── api
│ └── main.py
│
├── cache
│ └── semantic_cache.py
│
├── clustering
│ └── fuzzy_cluster.py
│
├── data
│ └── loader.py
│
├── embeddings
│ └── embedder.py
│
├── vectordb
│ └── vector_search.py
│
├── analysis
│ └── visualize_clusters.py
│
├── requirements.txt
└── README.md

## Installation

Clone the repository and install dependencies.


python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt



## Running the API

Start the FastAPI server:


uvicorn api.main:app --reload


The API will run at:


http://127.0.0.1:8000


API documentation is available at:


http://127.0.0.1:8000/docs



## API Endpoints

### POST /query

Submit a natural language query.

Example request:


{
"query": "space rocket launch"
}


Example response:


{
"query": "space rocket launch",
"cache_hit": false,
"result": [
{
"document": "NASA launched...",
"score": 0.82
}
],
"dominant_cluster": 3
}



### GET /cache/stats

Returns cache statistics.

Example response:


{
"total_entries": 12,
"hit_count": 4,
"miss_count": 8,
"hit_rate": 0.33
}



### DELETE /cache

Clears the semantic cache.


## Cluster Visualization

To visualize document clusters:


python -m analysis.visualize_clusters


This produces a 2D PCA visualization of document embeddings.


## Design Decisions

### Embedding Model
`all-MiniLM-L6-v2` was chosen because it provides high-quality semantic embeddings while remaining lightweight and efficient.

### Clustering
Gaussian Mixture Models were used to enable **fuzzy clustering**, allowing documents to belong to multiple clusters probabilistically.

### Semantic Cache
The cache compares query embeddings using cosine similarity. If similarity exceeds the threshold (0.85), the cached result is returned.

This avoids recomputing results for semantically similar queries.


## Technologies Used

- Python
- FastAPI
- SentenceTransformers
- scikit-learn
- FAISS-style vector similarity (cosine similarity)
- Matplotlib


## Future Improvements

- Dynamic cluster labeling
- More advanced vector database (FAISS indexing)
- Adaptive cache thresholds
- Query analytics and monitoring
