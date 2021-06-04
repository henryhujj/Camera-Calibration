# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:47:29 2020

@author: a0952
"""
# IMPORT NECESSARY LIBRARIES

import imageio
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import numpy as np
import cv2

class PCAFun():
    
    def comp_2d(self, image_2d): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
    	cov_mat = image_2d - np.mean(image_2d , axis = 1)
    	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
    	p = np.size(eig_vec, axis =1)
    	idx = np.argsort(eig_val)
    	idx = idx[::-1]
    	eig_vec = eig_vec[:,idx]
    	eig_val = eig_val[idx]
    	numpc = 80 # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
    	if numpc <p or numpc >0:
    		eig_vec = eig_vec[:, range(numpc)]
    	score = np.dot(eig_vec.T, cov_mat)
    	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
    	recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
    	return recon_img_mat
    def Fun(self):
        # IMPORTING IMAGE USING SCIPY AND TAKING R,G,B COMPONENTS
        fig = plt.figure(figsize=(50,50))
        cnt=0
        cnt2 = 0
        for i in range(1,35):
            cnt = i +17
            cnt2 = i+34
            if i <18:
                ax = fig.add_subplot(4, 17, i)
            else:
                ax = fig.add_subplot(4, 17, cnt)
            a = imageio.imread("%d.jpg"%(i))
            a_np = np.array(a)
            a_r = a_np[:,:,0]
            a_g = a_np[:,:,1]
            a_b = a_np[:,:,2]
            a_r_recon, a_g_recon, a_b_recon = self.comp_2d(a_r), self.comp_2d(a_g), self.comp_2d(a_b) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
            recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
            recon_color_img2 = Image.fromarray(recon_color_img)
            # recon_color_img.show()
            a_gray= cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
            final_gray = cv2.cvtColor(recon_color_img,cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(a_gray,final_gray)
            sum_ = 0
            for k in range(diff.shape[0]):
                for d in range(diff.shape[1]):
                    sum_ = sum_+diff[k][d] 
            print(sum_)
            ax.imshow(recon_color_img2)
            if i <18:
                ax = fig.add_subplot(4, 17, cnt)
            else:
                ax = fig.add_subplot(4, 17, cnt2)
            ax.imshow(a)
        plt.show()