#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:00:48 2021

@author: oscar
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

inputDir = "/home/oscar/data/biopsy/tiff/dataset_1/csv"

trainFileName = "train_dataset.csv"
testFileName = "test_dataset.csv"

trainDataSet = pd.read_csv(inputDir+"/"+trainFileName)
testDataSet = pd.read_csv(inputDir+"/"+testFileName)

trainX = trainDataSet.iloc[:,0:11]
trainY = trainDataSet.iloc[:,-1]

testX = testDataSet.iloc[:,0:11]
testY = testDataSet.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(trainX)
trainX = pd.DataFrame(scaler.transform(trainX)) # to pandas data frame

#scaler = StandardScaler()
scaler.fit(testX)
testX = pd.DataFrame(scaler.transform(testX)) # to pandas data frame



paramGrid = {'C': [1, 10, 100, 1000, 10000, 100000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'scale', 'auto'], }
estimator = GridSearchCV(svm.SVC(kernel='rbf'), paramGrid, scoring='accuracy', n_jobs=8)
estimator.fit(trainX, trainY);
    
print("Best estimator found by grid search:")
print(estimator.best_params_)

targetNames = ['Non_pleura', 'Pleura']
pred = estimator.predict(testX)
print(classification_report(testY, pred, target_names=targetNames))


