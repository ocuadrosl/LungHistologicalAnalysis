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



def balanceDateset(x, y):
    """
        Simple balancing datasets by removing random features
    """

    nPleura = sum(y == 1)
    nNonPleura = sum(y == -1)
    nRemove = abs(nPleura - nNonPleura);
    print(nRemove)

    # remove non pleura
    if nPleura < nNonPleura:
        print("Balancing non pleura")
        indices = np.array(np.where(y == -1)).ravel()
        #print(indices, "indices")
        removeIndices = np.random.choice(indices, nRemove, replace=False)
        x.drop(removeIndices)
        y.drop(removeIndices)
    else: # remove non pleura
        print("Balancing  pleura")
        indices = np.array(np.where(y == 1)).ravel()
        removeIndices = np.random.choice(indices, nRemove, replace=False)
        x = x.drop(removeIndices)
        y = y.drop(removeIndices)


def SplitDatasets(trainDataSet, testDataSet ):

    train_X = trainDataSet.iloc[:, 5:-1]
    train_Y = trainDataSet.iloc[:, -1]
    # print(train_Y)
    test_X = testDataSet.iloc[:, 5:-1]
    test_Y = testDataSet.iloc[:, -1]

    return train_X, train_Y, test_X, test_Y

def ScaleData(trainX, testX):
    scaler = StandardScaler()
    scaler.fit(trainX)
    # trainX = pd.DataFrame(scaler.transform(trainX))  # to pandas data frame
    trainX = scaler.transform(trainX)
    
    scaler2 = StandardScaler()
    scaler2.fit(testX)
    # testX = pd.DataFrame(scaler2.transform(testX))  # to pandas data frame
    testX = scaler2.transform(testX)  # to pandas data frame
    return trainX, testX


def SVM(trainX, trainY, testX, testY):
    
    paramGrid = {'C': [1, 10, 100, 1000, 10000, 100000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'scale', 'auto'], }
    estimator = GridSearchCV(svm.SVC(kernel='rbf'), paramGrid, scoring='accuracy', n_jobs=8)
    estimator.fit(trainX, trainY)
        
    print("Best estimator found by grid search:")
    print(estimator.best_params_)
    
    targetNames = ['Non_pleura', 'Pleura']
    classification = estimator.predict(testX)
    print(classification_report(testY, classification, target_names=targetNames))
    return estimator, classification




if __name__ == "__main__":
    
    inputDir = "/home/oscar/data/biopsy/tiff/dataset_2/csv"
    trainFileName = "train_erode_radius_30_LBP_3.csv"
    testFileName = "test_erode_radius_30_LBP_3.csv"

    trainDataSet = pd.read_csv(inputDir+"/"+trainFileName)
    testDataSet = pd.read_csv(inputDir+"/"+testFileName)

    train_X, train_Y, test_X, test_Y = SplitDatasets(trainDataSet, testDataSet)

    print("Pleura ", sum(train_Y == 1))
    print("Non pleura ", sum(train_Y == -1))
    # balanceDateset(train_X, train_Y)


    train_X, test_X = ScaleData(train_X, test_X)

    _, classification = SVM(train_X, train_Y, test_X, test_Y)

    # write the classification result
    classResult = testDataSet.iloc[:, 0:5]
    classResult['classification'] = pd.DataFrame(classification)
    classResult.to_csv(inputDir+"/classification_"+testFileName,  index=False)
