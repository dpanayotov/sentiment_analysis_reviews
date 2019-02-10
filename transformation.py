import re

import nltk

tokenizer = nltk.WhitespaceTokenizer()
stemmer = nltk.PorterStemmer()


class FitlessTransformer:
    def fit(self, X, y=None):
        return self


class TextTransformer(FitlessTransformer):
    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def transform(self, X):
        text = (" ".join(nltk.word_tokenize(row)) for row in X)
        if self.lowercase:
            return [x.lower() for x in text]
        return list(text)


class RegexTransformer(FitlessTransformer):
    def __init__(self, regex_pairs):
        self.regex_pairs = regex_pairs

    def transform(self, X):
        return [self.replace(x) for x in X]

    def replace(self, x):
        for regex, replacement in self.regex_pairs:
            x = re.sub(regex, replacement, x)
        return x


class StemmingTransformer(FitlessTransformer):

    def transform(self, X):
        result = []
        for x in X:
            tokens = nltk.word_tokenize(x)
            stemmed = [stemmer.stem(token) for token in tokens]
            result.append(' '.join(stemmed))
        return result


class CleanTransformer(FitlessTransformer):
    def __init__(self, stop_words):
        self.stop_words = stop_words

    def transform(self, X):
        result = []
        for x in X:
            tokens = nltk.word_tokenize(x)
            tagged = nltk.tag.pos_tag(tokens)
            tagged_tokens = [token.lower() for (token, tag) in tagged if
                             tag != 'NNP' and tag != 'NNPS']
            no_stopwords = [token for token in tagged_tokens if token not in self.stop_words]
            result.append(' '.join(no_stopwords))
        return result


class FinalTransformer(FitlessTransformer):

    def transform(self, X):
        result = []
        for x in X:
            tokens = nltk.word_tokenize(x)
            tagged_tokens = [token for token in tokens if len(token) > 2]
            result.append(' '.join(tagged_tokens))
        return result
