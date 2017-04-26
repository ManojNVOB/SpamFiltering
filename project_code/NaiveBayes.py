

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

df = pd.read_csv('/home/datascience/Desktop/SpamFiltering/features.csv', sep=',', lineterminator='\n',
                 na_values=['-'])

columns = list(df.columns.values)
words = columns[:]
words.remove("is_spam")
words.remove("file_name")
words.remove("ind")

df_spam = pd.DataFrame()
df_spam["is_spam"] = df["is_spam"]
df.drop('is_spam', axis=1, inplace=True)

df.drop('file_name', axis=1, inplace=True)
df.drop('ind', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df,df_spam, train_size = 0.6)


gaussianclf = GaussianNB()
gaussianclf.fit(X_train,y_train)
gaussianclfresult = gaussianclf.predict(X_test)
print "Gaussian NB results"
print "confusion matrix"
print confusion_matrix(y_test, gaussianclfresult)
print "precision_recall scores"
print precision_recall_fscore_support(y_test,gaussianclfresult,pos_label=0)


bernoulliclf = BernoulliNB()
bernoulliclf.fit(X_train,y_train)
bernoulliclfresult = bernoulliclf.predict(X_test)
print "Bernoulli NB results"
print "confusion matrix"
print confusion_matrix(y_test, bernoulliclfresult)
print precision_recall_fscore_support(y_test,bernoulliclfresult,pos_label=0)


multinomialclf = MultinomialNB()
multinomialclf.fit(X_train,y_train)
multinomialclfresult = multinomialclf.predict(X_test)
print "Multinomial NB results"
print "confusion matrix"
print confusion_matrix(y_test, multinomialclfresult)
print "precision_recall scores"
print precision_recall_fscore_support(y_test,result2,pos_label=0)

