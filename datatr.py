#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:18:23 2018

@author: ck807
"""

import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

#from keras.preprocessing.image import img_to_array, load_img

#def scaleRadius(img,scale):
#    x=img[int(img.shape[0]/2),:,:].sum(1)
#    r=(x>x.mean()/10).sum()/2
#    s=scale*1.0/r
#    return cv2.resize(img,(0,0),fx=s,fy=s)

#scale = 300
image_length = 256
image_height = 256
num_channels = 3
i = 0

data_file = glob.glob('/home/ck807/melenoma_seg/ISIC-2017_Training_Data/*.jpg')
files = []

data_file_mask = glob.glob('/home/ck807/melenoma_seg/ISIC-2017_Training_Part1_GroundTruth/*.png')

trainData = np.zeros((len(data_file),image_length, image_height, num_channels))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

trainLabel = np.zeros((len(data_file_mask),image_length,image_height,1))

for f in (data_file):
    a=cv2.imread(f)
#    a=scaleRadius(a,scale)
#    b=np.zeros(a.shape)
#    b=b.astype(np.uint8) 
#    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    resized_image = cv2.resize(a, (image_length, image_height))
    resized_image = resized_image.astype(np.float64)
    trainData[i,:,:,:] = resized_image[:,:,:]
    base = os.path.basename("/home/ck807/melenoma_seg/ISIC-2017_Training_Data/" + f)
    fileName = os.path.splitext(base)[0]
    files.append(fileName)
    i += 1
    
for k in (data_file_mask):
    base = os.path.basename("/home/ck807/melenoma_seg/ISIC-2017_Training_Part1_GroundTruth/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName[0:12]
    index = files.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    resized_image = cv2.resize(gray, (256, 256))
    gray_image = img_to_array(resized_image)
    trainLabel[index, :, :, :] = gray_image[:, :, :]
    
    
np.save('data.npy',trainData)
np.save('dataMask.npy', trainLabel)