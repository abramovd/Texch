from collections import defaultdict
from sklearn.cluster import spectral_clustering

from .base import BaseClusterer


class SpectralClustering(BaseClusterer):
    verbose_name = 'K means'
    cluster_class = spectral_clustering

    def find_clusters(self, data, *args, **kwargs):
        labels = self.cluster_class(data, *self.args, **self.kwargs)
        self._clusters = defaultdict(list)
        self._labels = labels
        for ind, cluster in enumerate(labels):
            self._clusters[cluster].append(ind)
        return self._clusters
