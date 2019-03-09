import keras
import os
import random
import warnings
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

from utility import *

# Ako se za keras kao backend koristi tensorflow onda se kao format koristi channels_last, 
# zbog toga sto tensorflow ocekuje da se RGB kanali navode kao poslednji argument, tj. ocekuje cetvorodimenzioni tensor oblika:
# (samples, rows, cols, channels)
K.set_image_data_format('channels_last')


# Da ne izbacuje warninge. Zakomentarisati po potrebi.
warnings.filterwarnings("ignore")

images, classes = load_data(train_data_dir)

NUM_CLASSES = len(set(classes))

print("Number of classes: {0}\nNumber of images for training: {1}".format(NUM_CLASSES, len(images)))


display_images_and_classes(images, classes)

#for image in images[:5]:
    #print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
    
    
# Postavljamo da sve slike budu istih dimenzija: 64x64
images64 = [skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE), mode='constant')
                for image in images]
#display_images_and_classes(images64, classes)

images = np.array(images64)
classes = np.array(classes)
#print(classes[3])

# Za klasu i postavljamo 1 na i-to mesto, a na ostala 0
classes = np.eye(NUM_CLASSES, dtype='uint8')[classes]
#print(classes[3])

print("classes shape: ", classes.shape, "\nimages shape: ", images.shape)


# TODO: Napraviti bolji model i/ili ga optimizovati drugacije!!!!!!!
def cnn_model():
    
    model = Sequential()

    # Konvolucija i sazimanje 1
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), data_format="channels_last", activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    # Konvolucija i sazimanje 2
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Konvolucija i sazimanje 3
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    # probati i ovo umesto onog dole
    """
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    """
    
    # Transformacija 3D slike u 1D vektor 
    model.add(Flatten())
    model.add(Dense(64)) # viseslojni perceptron
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # deaktiviramo polovinu neurona kako ne bi doslo do preprilagodjavanja
    model.add(Dense(NUM_CLASSES))  # output layer ima onoliko neurona koliko ima klasa
    model.add(Activation('softmax'))

    model.summary()
    
    return model



batch_size = 64
epochs = 20
lr = 0.01   #learning rate

model = cnn_model()

# optimizacija pomocu gradijentnog spusta
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

model.fit(images, classes,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)])