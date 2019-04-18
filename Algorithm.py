#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: sonalig
"""

import pandas as pd
from sklearn import preprocessing
import nltk
from sklearn.feature_extraction import text
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
import numpy as np
from sklearn.feature_selection import VarianceThreshold


## Read data into dataframes ##

df1 = pd.read_table('raw_tweets.txt', converters = {'Tweet ID' : str})
df2 = pd.read_table('ground_truth.txt', converters = {'Tweet ID': str})
raw_tweets = df1['Tweet Text']
orig_Labels = df2['Label']
le = preprocessing.LabelEncoder()
encodedLabels = le.fit_transform(orig_Labels)

## Preprocessing ##


processedText = raw_tweets.str.replace(r'£|\$', 'moneysymb')    
processedText = processedText.str.replace(r'[^\w\d\s]', ' ')
processedText = processedText.str.replace(r'\s+', ' ')
processedText = processedText.str.replace(r'^\s+|\s+?$', '')
processedText = processedText.str.lower()

###Stop word removal##

stop_words = nltk.corpus.stopwords.words('english')
processedText = processedText.apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words)))

### STEMMING ###

porter = nltk.PorterStemmer()
processedText = processedText.apply(lambda x: ' '.join(
        porter.stem(term) for term in x.split()))


## Feature Extraction : TF-IDF Vectorizer
vectorizer = text.TfidfVectorizer(ngram_range=(1,2))
X_ngrams = vectorizer.fit_transform(processedText)

features = []
for line in open('memos/Memo4.txt'):
    features.append([line])
    


X_train, X_test, y_train, y_test = train_test_split(X_ngrams,
                                                    encodedLabels,
                                                    test_size=0.2,
                                                    random_state = 42,
                                                    stratify=encodedLabels)

######## First round of classification

#1. Multinomial Naive Bayes#
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score_nb = metrics.f1_score(y_test, y_pred, average = 'micro')
#
print('\n\n')
print('NB Preliminary Analysis: Confusion Matrix')
print('-----------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual', 'actual'], ['against', 'for', 'neutral']],
             columns=[['predicted', 'predicted', 'predicted'], ['against', 'for', 'neutral']]))
print('\nNB Preliminary Analysis: Micro F1 Score')
print('----------------------------------')
print(score_nb)
print('\n')

#2. Logistic Regression
clf1 = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score_lr = metrics.f1_score(y_test, y_pred, average = 'micro')

print('\n\n')
print('LR Preliminary Analysis: Confusion Matrix')
print('-----------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual', 'actual'], ['against', 'for', 'neutral']],
             columns=[['predicted', 'predicted', 'predicted'], ['against', 'for', 'neutral']]))
print('\nLR Preliminary Analysis: Micro F1 Score')
print('----------------------------------')
print(score_lr)
print('\n')

# 10-fold validation
param_grid = [{'C': np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(estimator=LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial'),
                           param_grid=param_grid,
                           cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
                           scoring='f1_micro',
                           n_jobs=-1)

scores = cross_val_score(estimator=grid_search,
                         X=X_ngrams,
                         y=encodedLabels,
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                         scoring='f1_micro',
                         n_jobs=-1)


print('\n\n')
print('Validation Scores')
print('-----------------')
print(scores)
print('Mean Score', scores.mean())
print('\n\n')
######
#
# Final model: First round 
grid_search.fit(X_train, y_train)
valid_clf = LogisticRegression( C=grid_search.best_params_['C'], solver =  'newton-cg', multi_class = 'multinomial')
valid_clf.fit(X_train, y_train)
y_pred = valid_clf.predict(X_test)
test_error = metrics.f1_score(y_test, y_pred, average = 'micro')

print('\n\n')
print('LR First Round: Confusion Matrix')
print('-----------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual', 'actual'], ['against', 'for', 'neutral']],
             columns=[['predicted', 'predicted', 'predicted'], ['against', 'for', 'neutral']]))
print('\nLR Preliminary Analysis: Micro F1 Score')
print('----------------------------------')
print(test_error)
print('\n')


####################  Second round of classification


# use these probs as features 
probs_train = valid_clf.predict_proba(X_train)
probs_test = valid_clf.predict_proba(X_test)

sentiment_phrases = [x[0] for x in features]
sentiment_phrases = [x.strip() for x in sentiment_phrases]
sentiment_phrases = [x.split('@') for x in sentiment_phrases]

phrase_sentiment = []
for f in sentiment_phrases:
    if len(f) > 1:
        if f[1].strip() == 'positive':
            s=1;
        elif f[1].strip() == 'negative':
            s=2;
        phrase_sentiment.append([f[0], s])
        


phrases = [x[0] for x in sentiment_phrases]
phrases = [x.strip() for x in phrases]
phrases = [x.split('+') for x in phrases]

single_phrase = []
multiple_phrases = []

for x in phrases:
    if len(x) == 1:
        single_phrase.append(x)
    else:
        multiple_phrases.append(x)
        

Matrix = [[0 for x in range(58)] for y in range(400)]
phrase_count = np.array(Matrix)
Matrix = [[0 for x in range(17)] for y in range(400)]
co_count =  np.array(Matrix)
Matrix = [[0 for x in range(16)] for y in range(400)]
psent = np.array(Matrix)

## defining arrays and if these phrases and cophrases exist, give it 1
## if sentiment phrase occurs, give it sentiment 1,2
for i in range(len(processedText)):
    tweet = str(processedText[i])
    for j in range(len(single_phrase)):
       if(all(x in tweet for x in single_phrase[j])):
           phrase_count[i][j] = 1
    for k in range(len(multiple_phrases)):
        if(all(x in tweet for x in multiple_phrases[k])):
            co_count[i][k] = 1
    for l in range(len(phrase_sentiment)):
        if phrase_sentiment[l][0].strip() in tweet:
            psent[i][l] == phrase_sentiment[l][1]
            

probs_all = np.vstack((probs_train, probs_test))

#feature selection - remove features with no variance
Xnew_all = np.hstack((probs_all, phrase_count))
selector = VarianceThreshold()
Xnew_all = selector.fit_transform(Xnew_all)

#train test split
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xnew_all,
                                                    encodedLabels,
                                                    test_size=0.2,
                                                    random_state = 42,
                                                    stratify=encodedLabels)

 

#train on new training set

clfnew = LogisticRegression(solver = 'newton-cg' , multi_class = 'multinomial')
clfnew.fit(Xn_train, yn_train)
yn_pred = clfnew.predict(Xn_test)
score_lr_new = metrics.f1_score(yn_test, yn_pred, average = 'micro')


#10-fold validation

param_grid = [{'C': np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(estimator=LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial'),
                           param_grid=param_grid,
                           cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
                           scoring='f1_micro',
                           n_jobs=-1)

scores = cross_val_score(estimator=grid_search,
                         X=Xnew_all,
                         y=encodedLabels,
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                         scoring='f1_micro',
                         n_jobs=-1)


print('\n\n')
print('Validation Scores')
print('-----------------')
print(scores)
print('Mean Score', scores.mean())
print('\n\n')
#####
#
#
grid_search.fit(Xn_train, yn_train)
valid_clf = LogisticRegression( C=grid_search.best_params_['C'], solver = 'newton-cg', multi_class = 'multinomial')
valid_clf.fit(Xn_train, yn_train)
yn_pred = valid_clf.predict(Xn_test)
testn_error = metrics.f1_score(yn_test, yn_pred, average = 'micro')

print('\n\n')
print('LR Second Round: Confusion Matrix')
print('-----------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(yn_test, yn_pred),
             index=[['actual', 'actual', 'actual'], ['against', 'for', 'neutral']],
             columns=[['predicted', 'predicted', 'predicted'], ['against', 'for', 'neutral']]))
print('\nLR Preliminary Analysis: Micro F1 Score')
print('----------------------------------')
print(testn_error)
print('\n')