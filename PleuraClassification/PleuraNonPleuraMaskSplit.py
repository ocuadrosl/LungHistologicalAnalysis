#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:58:25 2021

@author: oscar

Split boundaries into Pleura and non-pleura masks
Require: manual labels 

"""
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    inputDir = "/home/oscar/data/biopsy/tiff/dataset_2"
    radius = 30
    masksDir = "erode_radius_"+str(radius)+"/"

    outputDirPleura = masksDir+"/pleura"
    outputDirPleura = masksDir+"/non_pleura"
    targetSet = 'test'

    print("START: " + targetSet)
    dataSet = None
    for imageName in os.listdir(inputDir+'/images_cleaned/'+targetSet):
        print(imageName)

        maskImage = cv2.imread(inputDir+"/boundary_masks/"+masksDir+"/original/"+imageName, cv2.IMREAD_GRAYSCALE)
        inputImage = cv2.imread(inputDir+'/images_cleaned/'+targetSet+'/'+imageName, cv2.IMREAD_GRAYSCALE)
        labelImage = cv2.imread(inputDir+'/labels/'+imageName, cv2.IMREAD_COLOR)
        if labelImage is None:
            print("Labeled Imaged", imageName, "does not exist")
            continue

        # Get pleural regions
        pleuraMask = np.zeros(maskImage.shape, np.uint8)
        pleuraMask[(labelImage == [0, 255, 4]).all(axis=2) * (maskImage == [255])] = [255]

        cv2.imwrite(inputDir+"/boundary_masks/"+masksDir+"/pleura/"+imageName, pleuraMask)

        # Get non pleural regions
        # dilate kernel to get non-pleura regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius*2, radius*2))
        pleuraMaskDilated = 255 - cv2.dilate(pleuraMask, kernel, iterations = 1)

        nonPleuraMask = (maskImage * pleuraMaskDilated)*255
        # eliminate small components from nonPleuraMask
        components, labels, stats, centroids  = cv2.connectedComponentsWithStats(nonPleuraMask, connectivity=8)
        minSize = 3000
        smallComponents = np.array(np.where(stats[:,4] < minSize)[0])
        for i in smallComponents:
            nonPleuraMask[labels == i] = 0

        cv2.imwrite(inputDir+"/boundary_masks/"+masksDir+"/non_pleura/"+imageName, nonPleuraMask)
    print("END: " + targetSet)