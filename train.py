import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K


# Ako se za keras kao backend koristi tensorflow onda se kao format koristi channels_last, 
# zbog toga sto tensorflow ocekuje da se RGB kanali navode kao poslednji argument, tj. ocekuje cetvorodimenzioni tensor oblika:
# (samples, rows, cols, channels)
K.set_image_data_format('channels_last')



def load_data(data_dir):
    
    classes = []
    images = []
    
    data_dir_abs_path = os.path.abspath(data_dir)
    #print(data_dir_abs_path)
    
    file_names = [f for f in os.listdir(data_dir)]
    for f in file_names:
        images.append(skimage.data.imread(os.path.join(data_dir_abs_path, f)))
        classes.append(int(f[:3]))  # prva tri karaktera u nazivu slike oznacavaju klasu
        
    return images, classes



train_data_dir = "./dataset/train"
test_data_dir = "./dataset/test"

images, classes = load_data(train_data_dir)

IMG_SIZE = 32  # 32x32 images
NUM_CLASSES = len(set(classes))

print("Number of classes: {0}\nNumber of images for training: {1}".format(NUM_CLASSES, len(images)))


def display_images_and_classes(images, labels):
    """ Prikazuje prvu sliku iz svake od klasa.
        Tako mozemo da vidimo koje sve znakove model moze da nauci """
        
    unique_labels = set(labels)
    plt.figure("Existing classes", figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Za svaku klasu uzima prvu sliku iz images koja joj pripada.
        # index vraca prvo pojavljivanje date klase u listi labels u kojoj se nalaze sve klase sa ponavljanjem
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # Slike prikazuje u vidu mreze od 8 redova x 8 kolona
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.axis('off')
        # naziv klase (broj slika u toj klasi)
        plt.title("Class {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.suptitle('Types of traffic signs in dataset')
    plt.show()

display_images_and_classes(images, classes)

for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))