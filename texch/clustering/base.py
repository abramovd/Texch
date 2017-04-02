import time

from texch.utils import already_run


class BaseClusterer(object):

    cluster_class = None
    verbose_name = None

    def __init__(self, *args, **kwargs):
        self._start_time, self._end_time = None, None
        self._clusters = None
        self._labels = None
        self.args = args
        self.kwargs = kwargs
        self.cluster_args = ()
        self.cluster_kwargs = {}
        self._is_run = False

    @property
    def is_run(self):
        return self._is_run

    def set_cluster_params(self, *args, **kwargs):
        self.cluster_args = args
        self.cluster_kwargs = kwargs

    def run(self, data):
        self._start_time = time.time()
        self._clusters = self.find_clusters(data, *self.cluster_args, **self.cluster_kwargs)
        self._end_time = time.time()
        return self._clusters

    @property
    def spent(self):
        return self._end_time - self._start_time

    def find_clusters(self, data, *args, **kwargs):
        raise NotImplementedError

    def get_clusters(self):
        return self._clusters

    def get_labels(self):
        return self._labels

    def __repr__(self):
        return self.verbose_name

    __str__ = __repr__
