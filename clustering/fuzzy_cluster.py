from sklearn.mixture import GaussianMixture

class FuzzyCluster:

    def __init__(self, n_clusters=12):

        self.model = GaussianMixture(
            n_components=n_clusters
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def dominant_cluster(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        return probs.argmax()