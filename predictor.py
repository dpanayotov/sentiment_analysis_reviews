
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

import transformation as t


class SentimentPredictor:
    def __init__(self, classifier=None, stopwords='english', regex_replacements=None, ngram=1):
        pipeline = [
            t.RegexTransformer(regex_replacements),
            t.CleanTransformer(stopwords),
            # t.StemmingTransformer(),
            t.FinalTransformer(),
            TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, sublinear_tf=False,
                            ngram_range=(1, ngram))]

        if classifier is None:
            classifier = SGDClassifier(random_state=42, class_weight='balanced', max_iter=100, tol=1e-3, alpha=1e-5,
                                       loss='modified_huber', n_jobs=-1, verbose=5, epsilon=1e-2,
                                       learning_rate='constant', eta0=0.01, penalty='elasticnet')

        self.pipeline = make_pipeline(*pipeline)
        self.classifier = classifier
        self.dupes = Duplicates()

    def fit(self, X, y=None):
        self.dupes.fit(X, y)
        Z = self.pipeline.fit_transform(X, y)
        self.classifier.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pipeline.transform(X)
        labels = self.classifier.predict(Z)
        for i, phrase in enumerate(X):
            label = self.dupes.get(phrase)
            if label is not None:
                labels[i] = label
        return labels


class Duplicates:
    def __init__(self):
        self.dupes = {}

    def fit(self, X, y):
        for phrase, clazz in zip(X, y):
            self.dupes[self.key_for_phrase(phrase)] = clazz

    def get(self, phrase):
        key = self.key_for_phrase(phrase)
        return self.dupes.get(key)

    def key_for_phrase(self, x):
        return ' '.join(x.lower().split())
