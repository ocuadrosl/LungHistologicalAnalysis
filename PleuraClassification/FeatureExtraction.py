#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:41:35 2021

@author: oscar
"""

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import local_binary_pattern

inputDir = "/home/oscar/data/biopsy/tiff/dataset_1/images_cleaned/train/"

dataSet = None
for imageName in os.listdir(inputDir):
    print(imageName)

    #imageName = 'B 2018 8484 C_1x.tiff'
    maskImage = cv2.imread('/home/oscar/data/biopsy/tiff/dataset_1/boundary_masks/erode_radius_20/'+imageName, cv2.IMREAD_GRAYSCALE)
    inputImage = cv2.imread('/home/oscar/data/biopsy/tiff/dataset_1/images_cleaned/train/'+imageName, cv2.IMREAD_GRAYSCALE)
    labelImage = cv2.imread('/home/oscar/data/biopsy/tiff/dataset_1/labels/'+imageName, cv2.IMREAD_COLOR)
    
    #Get pleural regions
    pleuraMask = np.zeros(maskImage.shape, dtype=bool)
    pleuraMask[(labelImage == [0,255,4]).all(axis=2) * (maskImage==[255]) ] = [True]
    
    
    
    #Get non pleural regions
    nonPleuraMask = np.zeros(maskImage.shape, np.uint8) #np.uint8 OpenCV uses it
    nonPleuraMask[(pleuraMask!=[True] * (maskImage==[255]))] = [255]
    #elminate small components from nonPleuraMask
    components, labels, stats, centroids  = cv2.connectedComponentsWithStats(nonPleuraMask, connectivity=4)
    minSize = 3000
    smallComponents = np.array(np.where(stats[:,4] < minSize)[0]) 
    for i in smallComponents:
        nonPleuraMask[ labels == i] = 0
    nonPleuraMask = (nonPleuraMask > 0)
    
    
    
    #local Binary Pattern
    radius = 1
    nPoints = 8 * radius
    nBins = np.arange(0, nPoints + 3)
    
    lbp = local_binary_pattern(inputImage, nPoints, radius, method='default')
    histogramPleura,_ = np.histogram(lbp[pleuraMask], bins=nBins, range=(0, nPoints + 2))
    histogramNonPleura,_ = np.histogram(lbp[nonPleuraMask], bins=nBins, range=(0, nPoints + 2))
    
    histogramPleura = np.append(histogramPleura, 1)
    histogramNonPleura = np.append(histogramNonPleura, -1)
    
    if dataSet is None:
        dataSet = np.vstack((histogramPleura, histogramNonPleura))
    else:
        dataSet = np.vstack( (dataSet, np.vstack((histogramPleura, histogramNonPleura)) ) )

    #print(dataSet)


dataSet = pd.DataFrame(data=dataSet)

dataSet.to_csv(inputDir+"/dataset.csv")



#plt.plot(histogramPleura)
#plt.plot(histogramNonPleura)
#plt.figure()

#lbp[pleuraMask==False]=0

#plt.imshow(lbp, cmap='gray')


