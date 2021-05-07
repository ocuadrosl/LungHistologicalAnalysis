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
import time
import sys


def SplitImage(image, tileSize):
    '''
        Split image into tiles of size tileSize
    '''

    height, width = image.shape
    # print(image.shape)

    tiles = []
    maxMultHeight = height - (height % tileSize)
    maxMultWidth = width - (width % tileSize)
    # print(maxMultHeight, maxMultWidth)

    for i in range(0, maxMultHeight, tileSize):
        for j in range(0, maxMultWidth, tileSize):
            # yield image[i:i+tileSize, j:j+tileSize]
            tiles.append(image[i:i + tileSize, j:j + tileSize])
            # print(image[i:i+tileSize, j:j+tileSize])

    # yield image[maxMultHeight:height, maxMultWidth:width]

    last_tile = image[maxMultHeight:height, maxMultWidth:width]
    if last_tile.shape[0] > 0 and last_tile.shape[1] > 0:
        tiles.append(last_tile)
    # else:
    #    print(last_tile.shape)
    # print(len(tiles))
    return tiles


def LBPHistogramToDataset(lbpTiles, maskTiles, isPluera):
    """
    Extract histograms from lbp tiles masking with pleura or non-pleura tile masks 
    """
    dataset = None

    for lbpTile, mTile in zip(lbpTiles, maskTiles):


        # if not True in mTile:
        if mTile.shape[0]*mTile.shape[1] <= 0:
            print("fuck ", mTile.shape[0]*mTile.shape[1])

        if np.sum(mTile == True) <= (mTile.shape[0]*mTile.shape[1] * 0.2):
            continue
        # print(np.sum(mTile == True))

        # compute the histogram
        histogramPleura, _ = np.histogram(lbpTile[mTile], bins=nBins, range=(0, nBins))

        # Add a label to the histogram and convert them in DataFrame
        histogramPleura = pd.DataFrame(np.append(histogramPleura, isPluera))

        # Add the image name
        histogramPleura.loc[-1] = imageName
        histogramPleura.index = histogramPleura.index + 1
        histogramPleura = histogramPleura.sort_index()
        # print(histogramPleura)

        if dataset is None:
            dataset = histogramPleura.T
        else:
            dataset = pd.concat([dataset, histogramPleura.T])

    return dataset


if __name__ == "__main__":

    inputDir = "/home/oscar/data/biopsy/tiff/dataset_1"
    boundaryDataSet = "erode_radius_20"
    targetSet = 'test'
    tile_size = 200  # tiles

    print("START: " + targetSet)

    dataset = None

    for imageName in os.listdir(inputDir + '/images_cleaned/' + targetSet):
        print(imageName)

        inputImage = cv2.imread(inputDir + '/images_cleaned/' + targetSet + '/' + imageName, cv2.IMREAD_GRAYSCALE)
        # get mask and convert them to boolean
        pleuraMask = cv2.imread(inputDir + "/boundary_masks/" + boundaryDataSet + "/pleura/" + imageName,
                                cv2.IMREAD_GRAYSCALE) > 0
        nonPleuraMask = cv2.imread(inputDir + "/boundary_masks/" + boundaryDataSet + "/non_pleura/" + imageName,
                                   cv2.IMREAD_GRAYSCALE) > 0

        # local Binary Pattern (LBP)
        radius = 3
        nPoints = 8 * radius

        # Compute LBP fro the whole input image
        lbp = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
        nBins = int(lbp.max() + 1)

        # split masks into tiles   
        pleuraTiles = SplitImage(pleuraMask, tile_size)
        nonPleuraTiles = SplitImage(nonPleuraMask, tile_size)
        lbpTiles = SplitImage(lbp, tile_size)

        pleuraDataset = LBPHistogramToDataset(lbpTiles, pleuraTiles, 1)
        nonPleuraDataset = LBPHistogramToDataset(lbpTiles, nonPleuraTiles, -1)

        if dataset is None:
            dataset = pd.concat([pleuraDataset, nonPleuraDataset])
        else:
            dataset = pd.concat([dataset, pleuraDataset, nonPleuraDataset])

    dataset = pd.DataFrame(data=dataset)

    dataset.to_csv(inputDir + "/csv/" + targetSet + "_" + boundaryDataSet + "_LBP_" + str(radius) + ".csv", index=False)
    print("END: " + targetSet)
