from __future__ import print_function

import operator
import sys
import numpy as np

from collections import defaultdict
from nltk.cluster.kmeans import KMeansClusterer as _NLTK_KMeans
from texch import distance as distances

from .base import BaseClusterer


class NLTKKMeans(_NLTK_KMeans):
    def __init__(self, num_means, distance,  *args, **kwargs):
        self.max_iterations = kwargs.pop('max_iterations', 100)
        self.converged = False
        self.num_iterations = 0
        super(NLTKKMeans, self).__init__(num_means, distance, *args, **kwargs)

    def _cluster_vectorspace(self, vectors, trace=False):
        if self._num_means < len(vectors):
            # perform k-means clustering
            converged = False
            self.num_iterations = 0
            while not converged and self.num_iterations < self.max_iterations:
                # assign the tokens to clusters based on minimum distance to
                # the cluster means
                clusters = [[] for m in range(self._num_means)]
                for vector in vectors:
                    index = self.classify_vectorspace(vector)
                    clusters[index].append(vector)

                # recalculate cluster means by computing the centroid of each cluster
                new_means = list(map(self._centroid, clusters, self._means))

                # measure the degree of change from the previous step for convergence
                difference = self._sum_distances(self._means, new_means)
                if difference < self._max_difference:
                    converged = True
                if trace:
                    print('iteration: {0}'.format(self.num_iterations))
                    print('difference: {0}'.format(difference))
                self.num_iterations += 1
                # remember the new means
                self._means = new_means
            if converged:
                self.converged = True


class NLTKKMeansObservationCentroids(NLTKKMeans):

    def _cluster_vectorspace(self, vectors, trace=False):
        if self._num_means < len(vectors):
            # perform k-means clustering
            converged = False
            self.num_iterations = 0
            #list_vectors = vectors.tolist()
            while not converged and self.num_iterations < self.max_iterations:
                # assign the tokens to clusters based on minimum distance to
                # the cluster means
                clusters = [[] for m in range(self._num_means)]
                for vector in vectors:
                    index = self.classify_vectorspace(vector)
                    clusters[index].append(vector)

                # recalculate cluster means by computing the centroid of each cluster
                new_means = list(map(self._centroid, clusters, self._means, vectors))

                # measure the degree of change from the previous step for convergence
                difference = self._sum_distances(self._means, new_means)
                if difference < self._max_difference:
                    converged = True
                if trace:
                    print('iteration: {0}'.format(self.num_iterations))
                    print('difference: {0}'.format(difference))
                self.num_iterations += 1
                # remember the new means
                self._means = new_means
            if converged:
                self.converged = True

    def _centroid(self, cluster, mean, vectors):
        if self._avoid_empty_clusters:
            dists = {}
            for i, vector in enumerate(cluster):
                dists[i] = 0
                for j, vector2 in enumerate(cluster):
                    if i != j:
                        dists[i] += self._distance(mean, vector)
                        dists[i] /= float(1 + len(cluster))
            centroid = max(dists.items(), key=operator.itemgetter(1))[0]
            return vectors[centroid]
        else:
            if not len(cluster):
                sys.stderr.write('Error: no centroid defined for empty cluster.\n')
                sys.stderr.write('Try setting argument \'avoid_empty_clusters\' to True\n')
                assert(False)
            dists = {}
            for i, vector in enumerate(cluster):
                dists[i] = 0
                for j, vector2 in enumerate(cluster):
                    if i != j:
                        dists[i] += self._distance(mean, vector)
                    dists[i] /= float(len(cluster))
            centroid = max(dists.items(), key=operator.itemgetter(1))[0]
            return vectors[centroid]


class KMeans(BaseClusterer):
    verbose_name = 'K means'
    cluster_class = NLTKKMeans

    def __init__(self, num_means, distance, *args, **kwargs):
        """

        :param dist:
        """
        if isinstance(distance, basestring):
            try:
                distance = getattr(distances, distance)
            except AttributeError:
                raise AttributeError(
                    'Unknown distance {0}. Choices: {1}'.format(
                        distance,
                        distances.names
                    )
                )
        elif callable(distance):
            pass
        else:
            raise AttributeError(
                'distance should be a scipy.spatial.distance str or callable'
            )
        super(KMeans, self).__init__(num_means, distance, *args, **kwargs)

    def find_clusters(self, data, *args, **kwargs):
        clusterer = self.cluster_class(*self.args, **self.kwargs)
        kwargs.setdefault('assign_clusters', True)
        labels = clusterer.cluster(
            data,
            *args,
            **kwargs
        )
        self.converged = clusterer.converged
        self.num_iterations = clusterer.num_iterations
        self._clusters = defaultdict(list)
        self._labels = labels
        for ind, cluster in enumerate(labels):
            self._clusters[cluster].append(ind)
        return self._clusters


class KMeansObservationAsCentroid(KMeans):
    verbose_name = 'K means with observations as centroids'
    cluster_class = NLTKKMeansObservationCentroids
