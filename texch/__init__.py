from __future__ import print_function

from sklearn.metrics.cluster import (
    entropy,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    silhouette_score
)
from texch.utils import already_run


class SingleExperiment(object):

    def __init__(self, data, clustering_algorithm, preprocessor=None,
                 true_labels=None, verbose_name=None, prepare_func=None):
        self.data = data
        self._true_labels = true_labels
        self.preprocessor = preprocessor
        self.clustering_algorithm = clustering_algorithm
        self.preprocessed_data = None
        self.preprocessed_features = None
        self.verbose_name = verbose_name or \
            'Preprocessor: ' + repr(preprocessor) + \
            '\nClustering algorithm: ' + repr(clustering_algorithm)
        self.prepare_func = prepare_func
        self._is_run = False

    def set_true_labels(self, true_labels):
        self._true_labels = true_labels

    @property
    def true_labels(self):
        return self._true_labels

    @property
    def spent(self):
        already_run(self)
        return {
            'preprocessor': self.preprocessor.spent,
            'clustering': self.clustering_algorithm.spent,
        }

    @property
    def total_spent(self):
        already_run(self)
        return sum(self.spent.values())

    @property
    def entropy(self):
        already_run(self)
        return entropy(self.get_labels())

    @property
    def homogeneity(self):
        already_run(self)
        if self.true_labels is None:
            print('To estimate quality firstly set true_labels on experiment')
        return homogeneity_score(self.true_labels, self.get_labels())

    @property
    def v_measure(self):
        already_run(self)
        if self.true_labels is None:
            print('To estimate quality firstly set true_labels on experiment')
        return v_measure_score(self.true_labels, self.get_labels())

    @property
    def adj_rand_index(self):
        already_run(self)
        if self.true_labels is None:
            print('To estimate quality firstly set true_labels on experiment')
        return adjusted_rand_score(self.true_labels, self.get_labels())

    @property
    def silhouette_coefficient(self):
        already_run(self)
        if self.true_labels is None:
            print('To estimate quality firstly set true_labels on experiment')
        return silhouette_score(self.preprocessed_data, self.get_labels())

    @property
    def completeness(self):
        already_run(self)
        if self.true_labels is None:
            print('To estimate quality firstly set true_labels on experiment')
        return completeness_score(self.true_labels, self.get_labels())

    def run_clustering(self):
        print('Running clustering...',)
        result = self.clustering_algorithm.run(self.preprocessed_data)
        print(self.clustering_algorithm.spent)
        return result

    @property
    def is_run(self):
        return self._is_run

    def run(self):
        print('Running experiment {0}'.format(self))
        self.run_preprocessing()
        if self.prepare_func is not None:
            print('Running in-middle prepare function')
            self.preprocessed_data = self.prepare_func(self.preprocessed_data)
        self._is_run = True
        self.run_clustering()

    def get_clusters(self):
        already_run(self)
        return self.clustering_algorithm.get_clusters()

    def get_labels(self):
        already_run(self)
        return self.clustering_algorithm.get_labels()

    def run_preprocessing(self):
        print('Running preprocessing...')
        self.preprocessor.set_input_data(self.data)
        self.preprocessor.run()
        self.preprocessed_data = self.preprocessor.result
        self.preprocessed_features = self.preprocessor.features

    def get_scores(self, scores='all'):

        if scores == 'all':
            scores = (
                'entropy', 'homogeneity',
                'completeness', 'adj_rand_index',
                # 'silhouette_coefficient',
                'v_measure'
            )

        score_results = {}
        for score in scores:
            if hasattr(self, score):
                score_results[score] = getattr(self, score)

        return score_results

    def summary(self, scores='all'):
        already_run(self)
        if self.true_labels is None:
            print('To print summary set true_labels on experiment')
            return
        labels = self.get_labels()
        clusters = self.get_clusters()
        print('Experiment ' + self.verbose_name + ' Summary')
        print('-------------------')
        print('Preprocessor:')
        print(self.preprocessor)
        print('Clustering algorithm:')
        print(self.clustering_algorithm)
        print('Total objects to cluster: {}'.format(len(labels)))
        print('Total clusters found: {0}'.format(len(clusters)))
        for num, cluster in clusters.items()[:5]:
            print('Cluster #{0}: {1} objects'.format(num, len(cluster)))
        print('.......')

        scores = self.get_scores(scores)
        print('Scores:')
        for name, value in scores.items():
            print('{0}: {1}'.format(name, value))

    def __repr__(self):
        return self.verbose_name or 'SingleExperiment'

    __str__ = __repr__


class MultiExperiment(object):

    def __init__(self, data, experiments, true_labels=None, name=None):
        self._experiments = []
        map(self.add_experiment, experiments)
        self.data = data
        self._true_labels = true_labels
        self._is_run = False
        self.verbose_name = name or ''

    @property
    def true_labels(self):
        return self._true_labels

    def add_experiment(self, experiment):
        if not isinstance(experiment, SingleExperiment):
            raise AttributeError(
                'experiment should be a subclass of SingleExperiment')
        self._experiments.append(experiment)

    def run(self):
        for experiment in self._experiments:
            experiment.run()
        self._is_run = True

    @property
    def spent(self):
        already_run(self)
        return {
            'Experiment #{0}: {1}'.format(num, repr(experiment)): experiment.spent
            for num, experiment in enumerate(self._experiments)
        }

    @property
    def total_spent(self):
        already_run(self)
        total = 0
        for experiment in self._experiments:
            total += experiment.total_spent
        return total

    @property
    def experiments(self):
        return self._experiments

    def set_true_labels(self, true_labels):
        self._true_labels = true_labels
        for experiment in self._experiments:
            experiment.set_true_labels(true_labels)

    def summary(self, scores='all'):
        already_run(self)
        if self.true_labels is None:
            print('Need to set true labels on MultiExperiment to print summary')
            return
        for num, experiment in enumerate(self._experiments):
            print('Experiment #{}'.format(num))
            experiment.summary(scores)
            print('------------------')
            print('------------------')

    def __repr__(self):
        return self.verbose_name or 'MultiExperiment'

    __str__ = __repr__
