import sys
import random

import numpy as np
import pandas as pd
from nltk import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from textblob import TextBlob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

import text_pre_processing as tpp


def _prepare_data():
    data_df = pd.read_csv('files/movie2.csv')
    data_df = data_df[:100]
    prepared_corpus = tpp.do_pre_processing(data_df['text'])
    data_df.insert(loc=2, column='clean text', value=prepared_corpus)
    data_df = data_df.replace(r'^(\s)+$', np.nan, regex=True)
    return data_df.dropna().reset_index(drop=True)


class Task1:
    def __init__(self):
        # prepare
        self.prepared_df = _prepare_data()
        # split to test and learn data
        self.learn_corpus, self.test_corpus, self.learn_label_names, self.test_label_names = train_test_split(
            self.prepared_df['clean text'],
            self.prepared_df['label'],
            test_size=0.3,
            random_state=0)

    def _part_a(self):
        # bag of words
        self.cv = CountVectorizer(min_df=0.0, max_df=1.0)
        cv_learn_features = self.cv.fit_transform(self.learn_corpus)
        cv_test_features = self.cv.transform(self.test_corpus)

        # svm
        self.svm = LinearSVC()
        self.svm.fit(cv_learn_features, self.learn_label_names)
        svm_test_score = self.svm.score(cv_test_features, self.test_label_names)
        svm_test_pred = self.svm.predict(cv_test_features)
        print('Confusion Matrix:')
        print(confusion_matrix(self.test_label_names, svm_test_pred))
        print('svm algorithm test accuracy: ', svm_test_score)

    def _analyze_sentiment_textblob(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity >= 0:
            return 1
        else:
            return 0

    def _part_b(self):

        test_pred_textblob = self.test_corpus.apply(self._analyze_sentiment_textblob)
        accuracy_textblob = accuracy_score(self.test_label_names, test_pred_textblob)
        conf_matrix_textblob = confusion_matrix(self.test_label_names, test_pred_textblob)
        print('Confusion Matrix:')
        print(conf_matrix_textblob)
        print('TextBlob accuracy:', accuracy_textblob)

    def _part_c(self):
        random_indexs = [random.randint(0, len(self.test_corpus)) for i in range(3)]
        for index in random_indexs:
            sent = self.test_corpus.iloc[index]
            true_sentiment = self.test_label_names.iloc[index]
            svm_prediction = self.svm.predict(self.cv.transform([sent]))[0]
            textblob_prediction = self._analyze_sentiment_textblob(sent)

            print("Sentence:", sent)
            print("True sentiment", true_sentiment)
            print("svm algorithm accuracy:", svm_prediction)
            print("TextBlob accuracy:", textblob_prediction)

    def do_task(self):
        self._part_a()
        self._part_b()
        self._part_c()
