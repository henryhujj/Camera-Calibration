# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:23:11 2021

@author: a0952
"""

import numpy as np
import cv2 as cv


def PT():
    cap = cv.VideoCapture('4perspective.mp4')
    im_src = cv.imread('rl.jpg')
    cnt = 1
    markerIds2 = np.zeros((4,1),dtype = 'float32')
    markerCorners2 = []
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
    while cv.waitKey(1) < 0:
    # try:
        pts_src = np.zeros((4,1,2),dtype='float32')
        pts_dst = np.zeros((4,1,2),dtype='float32')
        hasFrame, frame = cap.read()
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        parameters = cv.aruco.DetectorParameters_create()
        markerCorners,markerIds,rejectedCandiates = cv.aruco.detectMarkers(frame,dictionary,parameters=parameters)
        if len(markerIds)<4:
            markerIds = markerIds2
            markerCorners = markerCorners2
        markerIds2 = markerIds
        markerCorners2 = markerCorners
        #point 1
        index = np.squeeze(np.where(markerIds ==25))
        refPt1 = np.squeeze(np.array(markerCorners)[index[0]])[1]
        #point2
        index = np.squeeze(np.where(markerIds ==33))
        refPt2 = np.squeeze(np.array(markerCorners)[index[0]])[2]
        distance = np.linalg.norm(refPt1-refPt2)
        scalingFac = 0.2
        pts_dst[0] = [[refPt1[0]-round(scalingFac*distance),refPt1[1]-round(scalingFac*distance)]]
        pts_dst[1]= [[refPt2[0]+round(scalingFac*distance),refPt2[1]-round(scalingFac*distance)]]
        #point3
        index = np.squeeze(np.where(markerIds ==30))

        refPt3 = np.squeeze(np.array(markerCorners)[index[0]])[0]
        pts_dst[2]= [[refPt3[0]+round(scalingFac*distance),refPt3[1]+round(scalingFac*distance)]]
        #point4
        index = np.squeeze(np.where(markerIds ==23))
            

        refPt4 = np.squeeze(np.array(markerCorners)[index[0]])[0]
        pts_dst[3]= [[refPt4[0]-round(scalingFac*distance),refPt4[1]+round(scalingFac*distance)]]
        
        pts_src[0] =[0,0]
        pts_src[1] =[im_src.shape[1],0]
        pts_src[2] =[im_src.shape[1],im_src.shape[0]]
        pts_src[3] =[0,im_src.shape[0]]
        
        M, status = cv.findHomography(pts_src, pts_dst)
        im_dst = frame
        temp = cv.warpPerspective(im_src, M,(frame.shape[1],frame.shape[0]))
        # Prepare a mask representing region to copy from the warped image into the original frame.
        cv.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        im_dst = im_dst + temp  
        cv.namedWindow('final',cv.WINDOW_GUI_NORMAL)
        cv.imshow('final',im_dst)

        out.write(im_dst)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        cnt =cnt +1

    cap.release()         
    cv.destroyAllWindows()
