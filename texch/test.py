import numpy as np

from time import time
from sklearn import metrics

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from clustering import KMeans
from sklearn.cluster import KMeans as KM
from scipy.cluster.vq import kmeans

from random import Random

rd = Random(3)

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]

tfidf_vectorizer = TfidfVectorizer(max_df=0.999, max_features=200000,
                                   min_df=0.001, stop_words='english',
                                   use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset.data)
matrix = tfidf_matrix.todense()

dists = [
    # 'dice', - not converge
    # 'hamming',
    #  'kulsinski',
    # 'correlation'
    'euclidean',  'jaccard', 'cosine',
]

for dist in dists:
    rd = Random(42)
    clusterer = KMeans(true_k, distance=dist, rng=rd)
    t0 = time()
    result = clusterer.cluster(matrix)
    print dist
    print("done in %0.3fs" % (time() - t0))

    print('Entropy: {0}'.format(metrics.cluster.entropy(result)))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, result))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, result))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, result))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, result))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(tfidf_matrix, result, sample_size=1000))
    print '-------'
    print '-------'
    print '-------'
# cl = KM(true_k)
# cl.fit(tfidf_matrix)


# print '-----'
# print '-----'
# rd = Random(3)
# clusterer2 = KMeans(true_k, rng=rd)
# t0 = time()
# result2 = clusterer2.cluster(matrix)
# print("done in %0.3fs" % (time() - t0))
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, result2))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, result2))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, result2))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, result2))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(tfidf_matrix, result2, sample_size=1000))
#
# print '-----'
# print '-----'
# clusterer3 = KM(true_k, random_state=3)
# t0 = time()
# km = clusterer3.fit(tfidf_matrix)
# print("done in %0.3fs" % (time() - t0))
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(tfidf_matrix, km.labels_, sample_size=1000))
