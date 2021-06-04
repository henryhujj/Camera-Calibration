#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import cv2 as cv
import glob
from Tool import Tool

class Areality():
    
    
    def __init__(self):
        return

    def FUN(k):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        axis = np.float32([[1,1,0], [3,5,0], [5,1,0],[3,3,-3]]).reshape(-1,3)

        cv.namedWindow("output", cv.WINDOW_NORMAL)
        img = cv.imread('Image/' +str(k)+'.bmp')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = Tool.drawing(img,corners2,imgpts)
        cv.imshow('output',img)
        cv.waitKey(5)




