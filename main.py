import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB

import processing
from predictor import SentimentPredictor

replacements = [
    (" n't", "n't"),
    (" 's", "'s"),
    (" 'd", "'d"),
    (" 're", "'re"),
    ("-LRB-", "("),
    ("-RRB-", ")"),
    ('\W', ' '),
    ('\s+', ' '),
    (r'[0-9]+', ''),
]

if __name__ == '__main__':
    training = pd.read_csv('./train.tsv', sep='\t', header=0)
    X = training.Phrase
    Y = training.Sentiment

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2, shuffle=True)

    # classifier = LinearSVC(class_weight='balanced', random_state=42, C=1.0)
    # classifier = MultinomialNB(alpha=0.2, fit_prior=True)
    predictor = SentimentPredictor(regex_replacements=replacements, stopwords=processing.stop_words, ngram=3)
    predictor.fit(X_train, Y_train)
    prediction = predictor.predict(X_test)

    print('accuracy %s' % accuracy_score(Y_test, prediction))
    print(classification_report(Y_test, prediction))
    print(confusion_matrix(Y_test, prediction))
    print(f1_score(Y_test, prediction, average=None))
    # pickle.dump(predictor, open('final_model', 'wb'))