# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import glob
from skimage.feature import local_binary_pattern
#to save the SVM trained model:
from joblib import load
from sklearn.metrics import confusion_matrix



nRows = 6
mCols = 6

#Importing the file created earlier from SVM training
clf=load("C:/FACULTA/LICENTA/Licenta/jaffe database/cod/ExpressionsPCAPolynomialDegree7SVM.joblib")

#Importing the file containg PCA training values
pca=load("C:/FACULTA/LICENTA/Licenta/jaffe database/cod/PCASpace.joblib") 

#Creating the test matrix
TestULBPHist=np.zeros((115,361))

test_sample=1

#Performing LBP
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Anger/scaledimages_Anger/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    #Storing the concatenated histogram values in the test matrix     
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=1
    img_number +=1
    test_sample+=1        

path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Disgust/scaledimages_Disgust/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=2
    img_number +=1        
    test_sample+=1

path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Fear/scaledimages_Fear/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=3
    img_number +=1        
    test_sample+=1
    
    path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Happiness/scaledimages_Happiness/*.tiff"
    img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=4
    img_number +=1        
    test_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Neutral/scaledimages_Neutral/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=5
    img_number +=1        
    test_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Sadness/scaledimages_Sadness/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=6
    img_number +=1        
    test_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Surprise/scaledimages_Surprise/*.tiff"
img_number = 1

for file in glob.glob(path):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    
    lbp = local_binary_pattern(imgGray, 8, 1, 'uniform').astype(np.uint8)
    
    
    plt.imshow(lbp,'gray')
    plt.show

    # Dimensions of the image
    sizeX = lbp.shape[1]
    sizeY = lbp.shape[0]
    
    concat_lbp_hist=np.zeros(360)
    ind_patch=0;    
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = lbp[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
            cv2.imshow('rois'+str(i)+str(j), roi)
            hist=np.histogram(roi,10)
            concat_lbp_hist[ind_patch*10:ind_patch*10+10]=hist[0]
            ind_patch+=1
            
    TestULBPHist[test_sample-1,0:360]=concat_lbp_hist
    TestULBPHist[test_sample-1,360]=7
    img_number +=1        
    test_sample+=1

TestPCAofULBPHist=pca.transform(TestULBPHist[:,0:360])


ncomp=30

dec_label = clf.predict(TestPCAofULBPHist[:,0:ncomp])
accuracy_test=clf.score(TestPCAofULBPHist[:,0:ncomp], TestULBPHist[:,360])
print(accuracy_test)
conf_mat=confusion_matrix(TestULBPHist[:,360],dec_label)
print(conf_mat)
