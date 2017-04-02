import nltk

from texch.preprocessing.base import PreprocessStep


class TokensToText(PreprocessStep):
    verbose_name = 'TokensToText'

    def process(self):
        return [' '.join(tokens) for tokens in self.input_data]

'''
def tokenize(text, lower=True, stopwords=None, ):
    """
    Tokenizing and stemming of input text
    Requires nltk library
    Args:
        text - sentence (as a string)
    Return:
        List of stemmed tokens
    """
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    w = 0
    while w < len(tokens):
        if tokens[w].lower() in stopwords or tokens[w] == "'s":
            del tokens[w]
            w -= 1
        w += 1

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token.lower())

    if stem:
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    else:
        return filtered_tokens
'''