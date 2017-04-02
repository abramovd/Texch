import time

from texch.utils import already_run


class PreprocessStep(object):

    verbose_name = 'Base Preprocessing Step'

    def __init__(self, *args, **kwargs):
        self._input_data = None
        self._start_time, self._end_time = None, None
        self._input_features = None
        self._result = None
        self.features = None
        self.args = args
        self.kwargs = kwargs
        self._is_run = False

    def set_input_data(self, data):
        self._input_data = data

    def set_input_features(self, features):
        self._input_features = features

    @property
    def input_data(self):
        return self._input_data

    @property
    def is_run(self):
        return self._is_run

    @property
    def input_features(self):
        return self._input_features

    def get_features(self):
        already_run(self)
        return self.features

    @property
    def result(self):
        already_run(self)
        return self._result

    @property
    def spent(self):
        already_run(self)
        return self._end_time - self._start_time

    def run(self):
        self._start_time = time.time()
        self._result = self.process(*self.args, **self.kwargs)
        self._end_time = time.time()
        self._is_run = True
        return self._result

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.verbose_name

    __str__ = __repr__
