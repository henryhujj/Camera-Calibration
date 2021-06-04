# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:56:16 2021

@author: a0952
"""
from matplotlib import pyplot as plt
import cv2
def SDMFun():
	imgL = cv2.imread('Image/imL.png',0)
	imgR = cv2.imread('Image/imR.png',0)
	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	disparity = stereo.compute(imgL,imgR)
	cv2.namedWindow("output")   
	cv2.resizeWindow("output", 5, 255)
	cv2.imshow('output', disparity)
	plt.imshow(disparity,'gray')
	plt.show()