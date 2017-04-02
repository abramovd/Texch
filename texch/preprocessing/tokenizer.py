import nltk

from nltk.tokenize import WhitespaceTokenizer as _WhitespaceTokenizer
from nltk.util import ngrams
from nltk import FreqDist as _FreqDist

from texch.preprocessing import PreprocessStep


class TextToSentencesTokenizer(PreprocessStep):
    verbose_name = 'Sentence tokenizer'

    def process(self, language='english'):
        return [
            nltk.sent_tokenize(text, language)
            for text in self.input_data
        ]


class SentencesToWordsTokenizer(PreprocessStep):
    verbose_name = 'From sentences to words tokenizer'

    def process(self, language='english'):
        return [
            nltk.word_tokenize(sentence, language)
            for sentence in self.input_data
        ]


class TextToWordsTokenizer(PreprocessStep):
    verbose_name = 'From text to words tokenizer'

    def process(self, language='english'):
        return [
           nltk.word_tokenize(sent, language)
           for text in self.input_data
           for sent in nltk.sent_tokenize(text, language)
        ]


class WhitespaceTokenizer(PreprocessStep):
    verbose_name = 'White space tokenizer'

    def process(self, *args, **kwargs):
        return [_WhitespaceTokenizer().tokenize(text) for text in self.input_data]


class NGrams(PreprocessStep):
    verbose_name = 'N grams'

    def process(self, n):
        return [list(ngrams(tokens, n)) for tokens in self.input_data]


class FreqDist(PreprocessStep):
    verbose_name = 'FreqDist'

    def process(self, n):
        return [_FreqDist(tokens) for tokens in self.input_data]
