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
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from timeit import default_timer as timer

from utility import *

# Ako se za keras kao backend koristi tensorflow onda se kao format koristi channels_last, 
# zbog toga sto tensorflow ocekuje da se RGB kanali navode kao poslednji argument, tj. ocekuje cetvorodimenzioni tensor oblika:
# (samples, rows, cols, channels)
K.set_image_data_format('channels_last')


# Da ne izbacuje warninge. Zakomentarisati po potrebi.
warnings.filterwarnings("ignore")

images, classes = load_data(train_data_dir)

NUM_OF_CLASSES = len(set(classes))

print("Number of classes: {0}\nNumber of images for training: {1}".format(NUM_OF_CLASSES, len(images)))


# Da vidimo koje sve klase postoje u trening skupu podataka.
# Zakomentarisati ako nas ne zanima.
display_images_and_classes(images, classes)

#for image in images[:5]:
    #print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
    
    
# Postavljamo da sve slike budu istih dimenzija: IMG_SIZE x IMG_SIZE 
images_resized = [skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE), mode='constant')
                for image in images]
#display_images_and_classes(images_resized, classes)

images = np.array(images_resized)
classes = np.array(classes)
#print(classes[26])

# Za klasu i postavljamo 1 na i-to mesto, a na ostala 0
classes = np.eye(NUM_OF_CLASSES, dtype='uint8')[classes]
#print(classes[26])


#print("classes shape: ", classes.shape, "\nimages shape: ", images.shape)


def cnn_model():
    
    model = Sequential()
    ''''
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), data_format="channels_last", activation='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES, activation='softmax'))
    '''
    
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), data_format="channels_last"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=NUM_OF_CLASSES, activation = 'softmax'))
    
    
    model.summary()
    
    return model



batch_size = 32    # broj trening podataka u jednoj iteraciji
epochs = 20
lr = 0.01          # learning rate

timer_start = timer()

model = cnn_model()

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# optimizacija pomocu gradijentnog spusta
#sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

history = model.fit(images, classes,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)])

timer_end = timer()

elapsed_time = timer_end - timer_start
print("Elapsed time: {0} seconds".format(elapsed_time))

# izlistavanje svih podataka
#print(history.history.keys())

# preciznost (accuracy)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# podaci o promasaju (loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

