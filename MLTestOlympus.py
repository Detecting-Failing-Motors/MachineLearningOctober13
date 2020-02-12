#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:47:16 2019

@author: tbryan
"""
import os

from sklearn import model_selection
from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import welch
from detect_peaks import detect_peaks
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy #seems non beneficial
from scipy.signal import hilbert

#Import Machine Learning Libraries
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


directory = os.listdir(os.getcwd())
TrainingDataFile = "DELETE.csv"
for file in directory:
    if file == TrainingDataFile:
        dataset = pd.read_csv(file,header = 0,index_col = 0)

X = dataset.values[:,1:(dataset.shape[1]-1)]
Y = dataset.values[:,0]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)   


clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
Y_test_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_test_pred))

#Begin Selecting Features using Chi Square
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import RFE

"""
test = SelectKBest(score_func=chi2, k=4)
test.fit(X_train,Y_train)
print(test)
print(chi2(X_train,Y_train))
"""

clf = RandomForestClassifier(n_estimators=1000)
rfe = RFE(clf, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

#For rfe = 3