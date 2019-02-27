import keras
import os
import sys
import warnings
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

from utility import *

K.set_image_data_format('channels_last')

def test_all():
    
    test_images, test_classes = load_data(test_data_dir)
    display_images_and_classes(test_images, test_classes)

    test_images = [skimage.transform.resize(test_image, (IMG_SIZE, IMG_SIZE), mode='constant')
                    for test_image in test_images]

    test_classes = np.array(test_classes)
    test_images = np.array(test_images)

    #print("Classes shape: ", test_classes.shape, "\nImages shape: ", test_images.shape)

    
    model = load_model('./model.h5')
    y_pred = model.predict_classes(test_images)
    
    # preciznost racunamo kao broj_tacno_klasifikovanih / ukupan_broj_klasa
    acc = np.sum(y_pred == test_classes) / np.size(y_pred)
    print("Test accuracy = {}".format(acc))
    
    #print("Real classes: ", test_classes)
    #print("Real size: ", np.size(test_classes))
    #print("Predicted classes: ", y_pred)
    #print("Predicted size: ", np.size(y_pred))
    
    
def test_one():
    
    image_path = input("Enter image path:\n")
    abs_image_path = os.path.abspath(image_path)
    #print(abs_image_path)
    
    image = skimage.data.imread(abs_image_path)
    image = skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE), mode='constant')
    image = np.array(image)
    image = np.expand_dims(image, axis = 0)     # (64, 64, 3) --> (1, 64, 64, 3)  - dodajemo dimenziju koja predstavlja broj slika (ovde imamo samo jednu sliku kojoj predvidjamo klasu)
    #print("Image shape", image.shape)
    
    model = load_model('./model.h5')
    prediction = model.predict_classes(image)
    print("Predicted class: ", prediction[0])
    

def main():
    
    warnings.filterwarnings("ignore")
    
    if len(sys.argv) != 2:
        print("""Usage:
        For testing all images from test dataset: python3 test.py -all
        For predicting class of one input image:  python3 test.py -one""")
        sys.exit(0)
        
    if(sys.argv[1] == "-all"):
        test_all()
    elif(sys.argv[1] == "-one"):
        test_one()
    else:
        print("""Usage:
        For testing all images from test dataset: python3 test.py -all
        For predicting class of one input image:  python3 test.py -one""")
        sys.exit(0)
        
    

if __name__ == '__main__':
    main()