from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
import numpy as np
import glob

import os
import cv2

def cnn_algo():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=Image_size+[3])
    # vgg= VGG19(input_shape=Image_size + [3], weights='imagenet', include_top=False) # [3] means its an rgb channel
    return vgg

# create a model object
def input_model():
    model= Model(input= vgg.input, output= prediction)
    return model


from keras.preprocessing.image import ImageDataGenerator
## Done data Aggumentation
def aggumentation_training():


    train_datagen= ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                      zoom_range=0.2, horizontal_flip=True)
    training_set= train_datagen.flow_from_directory(cwd + '/train',
                                                    target_size=(224, 224), batch_size=32, class_mode='categorical')
    return  training_set


def aggumentation_testing():
    test_datagen= ImageDataGenerator(rescale= 1./255)

    test_set= test_datagen.flow_from_directory(cwd+'/test',
                                                target_size=(224,224), batch_size= 32,class_mode= 'categorical')
    return  test_set


if __name__=="__main__":

    Image_size = [224, 224]
    cwd= os.getcwd()

    #images_test = cv2.imread(cwd + '/test', cv2.IMREAD_GRAYSCALE)
    train_path = cwd + '/train'   #'/home/karma/projects_shambhavi/video_face/train'
    valid_path = cwd + '/test'

    vgg = cnn_algo()
    # vgg.summary()

    for layers in vgg.layers: # we are keeping the weights constant, it is already trained by the thousands of weights that is in imagenet
        layers.trainable = False  # we are not training all the layers

    x = Flatten()(vgg.output)  # need to add last layer, after flattning we can add the last layer
    folders = glob.glob(cwd + '/train/*' ) # check number of folders inside training folder
    print(folders)

    prediction = Dense(len(folders), activation='softmax')(x)# its the sigmoid activation function
    model= Dropout(0.2)
    model = input_model()
    model.summary()  # view the structure of model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_set = aggumentation_training()
    test_set = aggumentation_training()
    r = model.fit_generator(training_set, epochs=3, steps_per_epoch=len(training_set),
                            validation_steps=len(test_set))
    model.save('My_face_features_model.h5')
