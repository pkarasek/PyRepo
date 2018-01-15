import cv2
import sys
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

kernel = cv2.getGaussianKernel(347, 0)


img = cv2.imread('censor.jpg')

img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
censor_img = img.copy()
gray = cv2.cvtColor(censor_img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)
#print len(faces)

for face in faces:
    #print face
    x, y, w, h = face
    #cv2.rectangle(censor_img,(x,y),(x+w,y+h), (200,0,124), 2)
    face2censor = censor_img[y:y+h,x:x+w]
    face2censor = cv2.filter2D(face2censor, -1, kernel)
    censor_img[y:y+h,x:x+w] = face2censor
    
  
    

cv2.imshow('CENSORED', censor_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
