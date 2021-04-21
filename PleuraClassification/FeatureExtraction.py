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



inputDir = "/home/oscar/data/biopsy/tiff/dataset_1"
boundaryDataSet = "erode_radius_30"
targetSet = 'train'

print("START: "+ targetSet)
dataSet = None

for imageName in os.listdir(inputDir+'/images_cleaned/'+targetSet):
    print(imageName)
  
    inputImage = cv2.imread(inputDir+'/images_cleaned/'+targetSet+'/'+imageName, cv2.IMREAD_GRAYSCALE)
    #get mask and convert them to boolean
    pleuraMask = cv2.imread(inputDir+"/boundary_masks/"+boundaryDataSet+"/pleura/"+imageName, cv2.IMREAD_GRAYSCALE) > 0 
    nonPleuraMask = cv2.imread(inputDir+"/boundary_masks/"+boundaryDataSet+"/non_pleura/"+imageName, cv2.IMREAD_GRAYSCALE) > 0
   
    #cv2.imshow("image", np.asarray(nonPleuraMask*255, dtype="uint8"))
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows() 

    #local Binary Pattern
    radius = 5
    nPoints = 8 * radius
    #nBins = np.arange(0, nPoints + 3)
    
    lbp = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
    nBins = int(lbp.max() + 1)
    histogramPleura,_ = np.histogram(lbp[pleuraMask], bins=nBins, range=(0, nBins))
    histogramNonPleura,_ = np.histogram(lbp[nonPleuraMask], bins=nBins, range=(0, nBins))
    
    histogramPleura = np.append(histogramPleura, 1)
    histogramNonPleura = np.append(histogramNonPleura, -1)
    
    if dataSet is None:
        dataSet = np.vstack((histogramPleura, histogramNonPleura))
    else:
        dataSet = np.vstack( (dataSet, np.vstack((histogramPleura, histogramNonPleura)) ) )

    #print(dataSet)

    #lbp[pleuraMask==False]=0
    #plt.imshow(lbp, cmap='gray')
    #plt.waitforbuttonpress()
    

dataSet = pd.DataFrame(data=dataSet)

dataSet.to_csv(inputDir+"/csv/"+targetSet+"_"+boundaryDataSet+"_LBP_"+str(radius)+"_"+".csv")
print("END: "+ targetSet)
