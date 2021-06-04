# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:20:31 2021

@author: a0952
"""

import numpy as np
import cv2 as cv


def VD():
    cap = cv.VideoCapture('video.mp4')
    params = cv.SimpleBlobDetector_Params() 
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea =True
    params.minArea = 40
    params.maxArea = 60
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
    detector = cv.SimpleBlobDetector_create(params)
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    _,old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    keypoints = detector.detect(old_frame)
    p0 = np.zeros((7, 1, 2), dtype='float32')
    for i in range(len(keypoints)):
        p = np.array([[round(keypoints[i].pt[0]), round(keypoints[i].pt[1])]], dtype='float32')
        p0[i] = p
    mask = np.zeros_like(old_frame)
    while True:
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        for i,(new,old) in enumerate(zip(p1,p0)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv.line(mask, (a,b),(c,d), (0, 0, 255), 2)
                frame = cv.circle(frame,(a,b),5,(0, 0, 255),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = p1
    cv.destroyAllWindows()
    cap.release()