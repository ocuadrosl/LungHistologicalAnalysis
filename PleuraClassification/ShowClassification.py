#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:06 2021

@author: oscar
"""
import cv2
import pandas as pd
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def PaintTiles(image, mask, classification):
    # print(classification)

    for _, row in classification.iterrows():
        index = row.iloc[1:5]
        # print(index)
        maskTile = mask[index[0]:index[1], index[2]:index[3]]
        imageTile = image[index[0]:index[1], index[2]:index[3]]

        imageTile[np.where(maskTile == True)] = 0

    plt.imshow(image)
    plt.show()
    # input("Press Enter to continue...")


if __name__ == "__main__":
    # read classification results

    inputCSVPath = "/home/oscar/data/biopsy/tiff/dataset_1/csv"
    inputCSVFile = "classification_test_erode_radius_30_LBP_3.csv"
    dataset = pd.read_csv(inputCSVPath+"/"+inputCSVFile)

    # read images
    inputImagesPath = "/home/oscar/data/biopsy/tiff/dataset_1"
    targetSet = "images_cleaned/test"
    boundaryDataSet = "erode_radius_30"
    currentImage = ""
    for imageName in dataset.iloc[:, 0].drop_duplicates():

        print(imageName)
        inputImage = cv2.imread(inputImagesPath + "/" + targetSet+"/"+imageName, cv2.IMREAD_GRAYSCALE)

        # get mask and convert them to boolean
        pleuraMask = cv2.imread(inputImagesPath + "/boundary_masks/" + boundaryDataSet + "/pleura/" + imageName,
                                cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(inputImagesPath + "/boundary_masks/" + boundaryDataSet + "/non_pleura/" + imageName,
                                   cv2.IMREAD_GRAYSCALE) > 0

        print(type(pleuraMask))
        # passing only pleura for each image
        PaintTiles(inputImage, pleuraMask,  dataset[(dataset.iloc[:, 0] == imageName) & (dataset.iloc[:, -1] == 1)])




