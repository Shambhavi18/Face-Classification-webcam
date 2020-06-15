import cv2
import numpy as np
from os import listdir                   # list dir is used to fetch data from directory
from os.path import isfile, join
import os
face_classifier= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')# helps in cproping that perticular face
# function detect face and return crop face
# if no face detected then it return the input image
def face_extractor(img):
    faces= face_classifier.detectMultiScale(img,1.3, 5) # minsize and max size of the image
    if faces is ():
        return None
    # crop all faces found
    for(x,y,w,h) in faces:
        x=x-10
        y=y-10
        croped_face= img[y:y+h+50, x:x+w+50]
        return croped_face
# initialise webcam
cam= cv2.VideoCapture(0)
count= 0

cwd= os.getcwd()
# collect 50 samples of image from web cam
while True:
    ret,frame= cam.read()
    if face_extractor(frame) is not None:
        count= count+1
        face= cv2.resize(face_extractor(frame),(400,400))

        # save file
        file_name_path= cwd +'/test/0/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        # put count on iages
        cv2.putText(face,str(count),(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.imshow('Face cropper',face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count==100:
        break
cam.release()
cv2.destroyAllWindows()
print("collecting sample complete")




