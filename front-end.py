# Importing the libraries
from PIL import Image
# import pafy
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

model = load_model('My_face_features_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None
    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]
        return cropped_face



# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
# insert a function if wants to classify on youtube video
# def youtube_url():
#     # url = 'https://www.youtube.com/watch?v=bG1XV8tJaEM'
#     url = 'https://www.youtube.com/watch?v=hMy5za-m5Ew'
#     vPafy = pafy.new(url)
#     play = vPafy.getbest()
#     cap = cv2.VideoCapture(play.url)
#     return cap
# video_capture= youtube_url()

while True:
    _, img = video_capture.read()




    # image, face =face_detector(frame)

    face = face_extractor(img)
    # grey scale

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # im = cv2.imread(face,'RGB')
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        if (pred[0][0] > 0.54):
            name = 'Abhay'
            cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if (pred[0][1] > 0.95):
            name = 'Radha'
            cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


        # if (pred[0][2] > 0.5):
        #     name = 'Salman'
        #     cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #
        # if (pred[0][3] > 0.75):
        #     name = 'Shahrukh'
        #     cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(img, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()

cv2.destroyAllWindows()

