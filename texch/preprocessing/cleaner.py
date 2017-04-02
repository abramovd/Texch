from texch.preprocessing.base import PreprocessStep
from .constants import PUNCTUATION_REGEX

from nltk.corpus import stopwords as nltk_stopwords


class ClearPunctuationRegex(PreprocessStep):
    verbose_name = 'Clear punctuation'

    def process(self, regex=None):
        if regex is None:
            regex = PUNCTUATION_REGEX
        return [regex.sub('', text.strip()) for text in self.input_data]


class ExcludeChars(PreprocessStep):
    verbose_name = 'Exclude chars'

    def process(self, exclude):
        return [''.join(ch for ch in text if ch not in exclude) for text in self.input_data]


class LowerCase(PreprocessStep):
    verbose_name = 'To lower case'

    def process(self):
        return [text.lower() for text in self.input_data]


class RemoveStopWordsFromTokens(PreprocessStep):
    verbose_name = 'Remove stopwords from tokens'

    def process(self, stopwords=None, language='english'):
        if stopwords is None:
            # getting from nltk
            stopwords = nltk_stopwords.words(language)
        return [
            [
               token for token in tokens if token.lower() not in stopwords
            ]
            for tokens in self.input_data
        ]
