import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir + '/')

import text_pre_processing as tpp

# pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# prepare
data_df = pd.read_csv('files/science.csv')
data_df = data_df[:100]
data_df = data_df[~(data_df.Comment.str.strip() == '')]
prepared_corpus = tpp.do_pre_processing(data_df['Comment'])
data_df.insert(loc=2, column='Clean Comment', value=prepared_corpus)
data_df = data_df.replace(r'^(\s?)+$', np.nan, regex=True)
data_df = data_df.dropna().reset_index(drop=True)

print(data_df.head(10))

# split to test and learn data
learn_corpus, test_corpus, learn_label_names, test_label_names = train_test_split(data_df['Clean Comment'],
                                                                                  data_df['Topic'],
                                                                                  test_size=0.3,
                                                                                  random_state=0)

# bags of word
cv = CountVectorizer(min_df=0.0, max_df=1.0)
cv_learn_features = cv.fit_transform(learn_corpus)
cv_test_features = cv.transform(test_corpus)

# naive bayes algorithm
mnb = MultinomialNB()
mnb.fit(cv_learn_features, learn_label_names)
mnb_test_score = mnb.score(cv_test_features, test_label_names)
print('Naive Bayes algorithm test accuracy: ', mnb_test_score)

# svc algorithm
svm = LinearSVC()
svm.fit(cv_learn_features, learn_label_names)
svm_test_score = svm.score(cv_test_features, test_label_names)
print('Support vector machines algorithm test accuracy: ', svm_test_score)

# GridSearchCV update
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(cv_learn_features, learn_label_names)
best_svc_model = grid_search.best_estimator_
best_svc_pred = best_svc_model.predict(cv_learn_features)
best_svc_accuracy = accuracy_score(learn_label_names, best_svc_pred)

print("Best Accuracy of Support Vector Classifier after GridSearchCV:", best_svc_accuracy)


