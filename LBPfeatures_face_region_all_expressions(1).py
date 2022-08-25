# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import glob
from skimage.feature import local_binary_pattern
from sklearn import svm
#to save the SVM trained model:
from joblib import dump, load



nRows = 6
mCols = 6

#Creating the training matrix
TrainingULBPHist=np.zeros((91,361))

train_sample=1

#Applying LBP method for each image correspondiong to every emotion
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Anger/scaledimages_Anger/TrainImages/*.tiff"
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
            
    #The concatenated values of the histograms are stored in the training matrix    
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=1
    img_number +=1
    train_sample+=1        

path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Disgust/scaledimages_Disgust/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=2
    img_number +=1        
    train_sample+=1

path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Fear/scaledimages_Fear/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=3
    img_number +=1        
    train_sample+=1
    
    path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Happiness/scaledimages_Happiness/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=4
    img_number +=1        
    train_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Neutral/scaledimages_Neutral/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=5
    img_number +=1        
    train_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Sadness/scaledimages_Sadness/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=6
    img_number +=1        
    train_sample+=1
    
path = "C:/FACULTA/LICENTA/Licenta/jaffe database/cod/Surprise/scaledimages_Surprise/TrainImages/*.tiff"
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
            
    TrainingULBPHist[train_sample-1,0:360]=concat_lbp_hist
    TrainingULBPHist[train_sample-1,360]=7
    img_number +=1        
    train_sample+=1

#Applying the SVM classifier
clf = svm.SVC(kernel='linear',gamma='auto',decision_function_shape='ovo')
clf.fit(TrainingULBPHist[:,0:360], TrainingULBPHist[:,360])
dec_label = clf.predict(TrainingULBPHist[:,0:360])
accuracy_train=clf.score(TrainingULBPHist[:,0:360], TrainingULBPHist[:,360])
print(accuracy_train)
#Creating a file for the training data
dump(clf, 'ExpressionsLinearSVM.joblib') 