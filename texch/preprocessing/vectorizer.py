from sklearn.feature_extraction.text import (
    CountVectorizer as _CountVectorizer,
    TfidfTransformer as _TfidfTransformer,
    TfidfVectorizer as _TfidfVectorizer,
    HashingVectorizer as _HashingVectorizer
)

from texch.preprocessing import PreprocessStep


class BaseVectorizer(PreprocessStep):
    verbose_name = 'BaseVectorizer'
    vectrizer_class = None


class CountVectorizer(BaseVectorizer):
    verbose_name = 'CountVectorizer'
    vectorizer_class = _CountVectorizer

    def process(self, *args, **kwargs):
        vectorizer = self.vectorizer_class(*args, **kwargs)
        result = vectorizer.fit_transform(self.input_data)
        self.features = vectorizer.get_feature_names()
        return result.toarray()


class TfidfTransformer(BaseVectorizer):
    verbose_name = 'TfidfTransformer'
    vectorizer_class = _TfidfTransformer

    def process(self, *args, **kwargs):
        vectorizer = self.vectorizer_class(*args, **kwargs)
        result = vectorizer.fit_transform(self.input_data)
        return result


class TfidfVectorizer(BaseVectorizer):
    verbose_name = 'TfidfVectorizer'
    vectorizer_class = _TfidfVectorizer

    def process(self, *args, **kwargs):
        vectorizer = self.vectorizer_class(*args, **kwargs)
        result = vectorizer.fit_transform(self.input_data)
        self.features = vectorizer.get_feature_names()
        return result


class HashingVectorizer(BaseVectorizer):
    verbose_name = 'HashingVectorizer'
    vectorizer_class = _HashingVectorizer

    def process(self, *args, **kwargs):
        vectorizer = self.vectorizer_class(*args, **kwargs)
        result = vectorizer.fit_transform(self.input_data)
        return result
