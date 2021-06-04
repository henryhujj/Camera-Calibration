# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:18:39 2021

@author: a0952
"""

import numpy as np
import cv2 as cv

def BS():
    cap = cv.VideoCapture('bgsub.mp4')
    cnt = 0
    pixel_value = np.zeros((200,400,400), dtype = np.double)
    while cnt <=50:
        _,frame = cap.read()
        frame_gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        for i in range(frame_gray.shape[0]):
            for j in range(frame_gray.shape[1]):
                pixel_value[i,j,cnt] = frame_gray[i][j]
        if cv.waitKey(10) & 0xFF == 27:
            break
        cnt = cnt +1
    cap.release()
    mu = np.zeros((176,320))
    std = np.zeros((176,320))
    temp = np.zeros(51)
    for a in range(0,176,1):
        for b in range(0,320,1):
                for c in range(0,51,1):
                    temp[c]= pixel_value[a,b,c]
                    temp = np.zeros(51)
                mu[a,b] = np.mean(temp)
                std[a,b] = np.std(temp)
                if std[a,b]<5:
                    std[a,b] = 5
    cap2 = cv.VideoCapture('bgsub.mp4')
    diff = np.zeros((176,320))
    cv.namedWindow('final',cv.WINDOW_NORMAL)  
    while True:
        _,frame2 = cap2.read()
        frame2_gray = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        for i in range(0,176,1):
            for j in range(0,320,1):
                diff = frame2_gray[i][j]-mu[i,j]
                
                if abs(diff)>18*std[i,j]:           
                    frame2_gray[i][j] = 0
                else:
                    frame2_gray[i][j] = 255
    
        cv.imshow('final',frame2_gray)
        if cv.waitKey(1) & 0xFF == 27:
                    break    
    cap2.release()         
    cv.destroyAllWindows()