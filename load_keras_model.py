from imutils import paths
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,Activation,Flatten,Dropout
from keras import backend as K
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import pickle
import os
import datetime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer



def build(width,height,depth,classes,act="softmax"):
    model=Sequential()
    inputShape=(height,width,depth)
    chanDim=-1
    if K.image_data_format()=="channels_first":
        inputShape=(depth,height,width)
        chanDim=1
    ## conv-relu-pool
    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    ##conv-relu *2 ,pool
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    ## conv-relu*2,pool
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    ## flatten

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    ##
    model.add(Dense(classes))
    model.add(Activation(act))

    return model
