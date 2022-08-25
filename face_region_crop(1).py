# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import cv2 
import matplotlib.pyplot as plt
import glob




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Applying face detection using Viola-Jones algorithm and resizing the images for every emotion 
path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Anger/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Anger/scaledimages_Anger/imageAnger"+str(img_number)+".tiff",imgResized)
    img_number +=1
    
path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Disgust/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Disgust/scaledimages_Disgust/imageDisgust"+str(img_number)+".tiff",imgResized)
    img_number +=1    

path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Fear/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Fear/scaledimages_Fear/imageFear"+str(img_number)+".tiff",imgResized)
    img_number +=1

path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Happiness/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Happiness/scaledimages_Happiness/imageHappiness"+str(img_number)+".tiff",imgResized)
    img_number +=1

path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Neutral/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Neutral/scaledimages_Neutral/imageNeutral"+str(img_number)+".tiff",imgResized)
    img_number +=1    
   
path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Sadness/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Sadness/scaledimages_Sadness/imageSadness"+str(img_number)+".tiff",imgResized)
    img_number +=1   

path1 = "C:/FACULTA/LICENTA/Licenta/jaffe database/Surprise/*.tiff"
img_number = 1

for file in glob.glob(path1):
    imgFile=cv2.imread(file)
    
    imgGray = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        cv2.rectangle(imgGray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = imgGray[y:y+h, x:x+w]
        plt.imshow(imgGray,'gray')
        plt.show
        plt.imshow(roi_gray,'gray')
        plt.show
        imgResized=cv2.resize(roi_gray,(162,162),interpolation=cv2.INTER_AREA)
    
    cv2.imwrite("C:/FACULTA/LICENTA/Licenta/jaffe database/Surprise/scaledimages_Surprise/imageSurprise"+str(img_number)+".tiff",imgResized)
    img_number +=1    
    

  
   
   

    
    



