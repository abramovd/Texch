from nltk.stem.porter import PorterStemmer as _PorterStemmer
from nltk.stem.lancaster import LancasterStemmer as _LancasterStemmer
from nltk.stem.snowball import SnowballStemmer as _SnowballStemmer

from texch.preprocessing.base import PreprocessStep


class BaseStemmer(PreprocessStep):
    stemmer_class = None

    def process(self, *args, **kwargs):
        if self.stemmer_class is None:
            raise NotImplementedError('Need to specify stemmer class')
        stemmer = self.stemmer_class(*args, **kwargs)
        return [list(map(stemmer.stem, tokens)) for tokens in self.input_data]


class LancasterStemmer(PreprocessStep):
    verbose_name = 'Lancaster Stemmer'
    stemmer_class = _LancasterStemmer


class SnowballStemmer(PreprocessStep):
    verbose_name = 'Snowball Stemmer'
    stemmer_class = _SnowballStemmer


class PorterStemmer(PreprocessStep):
    verbose_name = 'Porter Stemmer'
    stemmer_class = _PorterStemmer
