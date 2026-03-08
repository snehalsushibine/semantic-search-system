import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorSearch:

    def __init__(self, embeddings, documents):

        self.embeddings = embeddings
        self.documents = documents


    def search(self, query_embedding, top_k=3):

        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({

                "document": self.documents[idx][:300],
                "score": float(similarities[idx])

            })

        return results