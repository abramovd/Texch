from nltk.corpus.reader.wordnet import NOUN
from nltk.stem.wordnet import WordNetLemmatizer as _WordNetLemmatizer

from texch.preprocessing.base import PreprocessStep


class WordNetLemmatizer(PreprocessStep):
    lemmatizer_class = _WordNetLemmatizer

    def process(self, pos=NOUN):
        lemmatizer = self.lemmatizer_class()
        return [
            [lemmatizer.lemmatize(token, pos) for token in tokens]
            for tokens in self.input_data
        ]
