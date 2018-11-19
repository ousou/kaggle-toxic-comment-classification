import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_reader import read_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin


class ToxicCommentBayesianEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.toxic_model = None
        self.severe_toxic_model = None
        self.obscene_model = None
        self.threat_model = None
        self.insult_model = None
        self.identity_hate_model = None
        self.count_vect = None
        self.tfidf_transformer = None

    def fit(self, X, y):
        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(X["comment_text"])
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        self.toxic_model = MultinomialNB().fit(X_train_tfidf, y["toxic"])
        self.severe_toxic_model = MultinomialNB().fit(X_train_tfidf, y["severe_toxic"])
        self.obscene_model = MultinomialNB().fit(X_train_tfidf, y["obscene"])
        self.threat_model = MultinomialNB().fit(X_train_tfidf, y["threat"])
        self.insult_model = MultinomialNB().fit(X_train_tfidf, y["insult"])
        self.identity_hate_model = MultinomialNB().fit(X_train_tfidf, y["identity_hate"])
        return self

    def predict(self, X, y=None):
        X_new_counts = self.count_vect.transform(X["comment_text"])
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        toxic_pred = self.toxic_model.predict(X_new_tfidf)
        severe_toxic_pred = self.severe_toxic_model.predict(X_new_tfidf)
        obscebe_pred = self.obscene_model.predict(X_new_tfidf)
        threat_pred = self.threat_model.predict(X_new_tfidf)
        insult_pred = self.insult_model.predict(X_new_tfidf)
        identity_hate_pred = self.identity_hate_model.predict(X_new_tfidf)
        return np.transpose(np.stack((X["id"], toxic_pred, severe_toxic_pred, obscebe_pred,
                         threat_pred, insult_pred, identity_hate_pred)))

    def predict_proba(self, X, y=None):
        X_new_counts = self.count_vect.transform(X["comment_text"])
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        toxic_pred = self.toxic_model.predict_proba(X_new_tfidf)[:,1]
        severe_toxic_pred = self.severe_toxic_model.predict_proba(X_new_tfidf)[:,1]
        obscebe_pred = self.obscene_model.predict_proba(X_new_tfidf)[:,1]
        threat_pred = self.threat_model.predict_proba(X_new_tfidf)[:,1]
        insult_pred = self.insult_model.predict_proba(X_new_tfidf)[:,1]
        identity_hate_pred = self.identity_hate_model.predict_proba(X_new_tfidf)[:,1]
        return np.transpose(np.stack((X["id"], toxic_pred, severe_toxic_pred, obscebe_pred,
                         threat_pred, insult_pred, identity_hate_pred)))


def fit_and_eval_bayes_model():
    X_train, y_train, X_test, y_test = read_data()
    bayes_model = ToxicCommentBayesianEstimator().fit(X_train, y_train)
    predictions = bayes_model.predict(X_test)
    print(y_test)
    print(predictions.shape)
    print(predictions)
    toxic_correct_preds = 0
    for index, row in y_test.iterrows():
        if row["toxic"] == predictions[index][1]:
            toxic_correct_preds += 1
    print("Toxic correct predictions: %i out of %i" % (toxic_correct_preds, predictions.shape[0]))




if __name__ == '__main__':
    fit_and_eval_bayes_model()
