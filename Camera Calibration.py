# -*- coding: utf-8 -*-
#use opencv to implement Find the chessboard Corner
"""
Created on Tue Jun  1 10:15:23 2021

@author: a0952
"""
import numpy as np
import cv2 as cv
import glob


img = cv.imread("your image")
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (11,8), None)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (11,8), corners2, ret)
		# calibration

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


rvecs = np.array(rvecs).reshape(3,1)
tvecs = np.array(tvecs).reshape(3,1)

Rmat = cv.Rodrigues(rvecs)[0].reshape(3,3)


cv.namedWindow("output", cv.WINDOW_NORMAL)
cv.imshow("output", img)
cv.waitKey(0)
