from __future__ import print_function

import time

from texch.preprocessing.base import PreprocessStep
from texch.exceptions import PreprocessingException, InputDataException
from texch.utils import already_run


class Preprocessor(object):

    def __init__(self, steps, name=None):
        self._steps = []
        map(self.add_step, steps)
        self._result = None
        self._data = None
        self.features = None
        self._is_run = False
        self.verbose_name = name

    def set_input_data(self, data):
        self._data = data

    @property
    def input_data(self):
        return self._data

    @property
    def is_run(self):
        return self._is_run

    def add_step(self, preprocessing_step):
        if not isinstance(preprocessing_step, PreprocessStep) \
                or hasattr(preprocessing_step, 'fit_transform'):
            raise AttributeError(
                'preprocessing_step should be a subclass of PreprocessStep or'
                ' has a method fit_transform'
            )
        self._steps.append(preprocessing_step)

    @property
    def steps(self):
        return self._steps

    def list_steps(self):
        steps = []
        for num, step in enumerate(self._steps):
            steps.append(
                'Step #{0}: {1}'.format(
                    num,
                    repr(step)
                )
            )
        return steps

    @property
    def spent(self):
        already_run(self)
        return {
            'Step#{0}: {1}'.format(num, repr(step)): step.spent
            for num, step in enumerate(self._steps)
        }

    @property
    def total_spent(self):
        already_run(self)
        total = 0
        for step in self._steps:
            total += step.spent
        return total

    def run(self):
        if self._data is None:
            raise InputDataException('Need to set_data for preprocessing')
        data = self._data
        features = self.features
        for num, step in enumerate(self._steps):
            try:
                if isinstance(step, PreprocessStep):
                    step.set_input_data(data)
                    step.set_input_features(features)
                    data = step.run()
                    print('Step#{0}: {1}: {2} sec'.format(
                            num,
                            repr(step),
                            step.spent
                        )
                    )
                    features = step.get_features()
                else:
                    start_time = time.time()
                    data = step.fit_transform()
                    setattr(step, 'spent', time.time() - start_time)
                    if hasattr(step, 'get_feature_names'):
                        features = step.get_feature_names()

            except Exception as err:
                raise PreprocessingException(
                    'Error during preprocesing step #{0}. {1}: {2}'.format(
                        num + 1,
                        step,
                        repr(err)
                    )
                )
        self._is_run = True
        self._result = data
        self.features = features
        return self._result

    @property
    def result(self):
        return self._result

    def __repr__(self):
        return self.verbose_name or '\n'.join(self.list_steps())

    __str__ = __repr__
