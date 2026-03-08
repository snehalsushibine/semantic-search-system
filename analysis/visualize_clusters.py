from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data.loader import load_dataset
from embeddings.embedder import Embedder

docs = load_dataset()

embedder = Embedder()

embeddings = embedder.embed(docs[:500])

pca = PCA(n_components=2)

reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8,6))

plt.scatter(
    reduced[:,0],
    reduced[:,1],
    alpha=0.5
)

plt.title("Semantic Document Clusters")

plt.xlabel("Component 1")

plt.ylabel("Component 2")

plt.show()