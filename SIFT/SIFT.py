# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:03:57 2021

@author: a0952
"""

import cv2

def Keypoint():
    img = cv2.imread('Image/Aerial1.jpg')
    img2 = cv2.imread('Image/Aerial2.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(6,None)
    kp = sift.detect(gray,None)
    kp2 = sift.detect(gray2,None)
    img=cv2.drawKeypoints(gray,kp,img)
    img2=cv2.drawKeypoints(gray2,kp2,img2)
    cv2.imwrite('FeatureAerial1.jpg',img)
    cv2.imwrite('FeatureAerial2.jpg',img2)

def Match():
    img1 = cv2.imread('Image/Aerial1.jpg')
    img2 = cv2.imread('Image/Aerial2.jpg')
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(45,None)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    img1=cv2.drawKeypoints(gray1,kp1,img1)
    img2=cv2.drawKeypoints(gray2,kp2,img2)
    cv2.imwrite('FeatureAerial1.jpg',img1)
    cv2.imwrite('FeatureAerial2.jpg',img2)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    cv2.imwrite('Figure2.jpg',img3)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", img3)
    cv2.waitKey(0)
