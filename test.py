import keras
import os, sys,json
import warnings
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import sklearn.metrics

from utility import *

K.set_image_data_format('channels_last')

def test_all():
    # Funkcija kojem se testiraju sve slike iz test skupa podataka.
    # Sluzi nam da vidimo kolika je preciznost napravljenog modela.
    
    # Ucitavamo slike i njihove klase iz skupa podataka za testiranje
    test_images, test_classes = load_data(test_data_dir)
    # Prikazujemo koje se sve klase znakova nalze u test skupu, i po jednu sliku za svaku od klasa
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
    print("\nTest accuracy = {}".format(acc))
    
    test_rep = sklearn.metrics.classification_report(test_classes, y_pred)
    print("\nClassification report:\n{}".format(test_rep))
    
    conf_matrix = sklearn.metrics.confusion_matrix(test_classes, y_pred)
    print("Confusion matrix:")
    print(conf_matrix)
        
    #print("Real classes: ", test_classes)
    #print("Real size: ", np.size(test_classes))
    #print("Predicted classes: ", y_pred)
    #print("Predicted size: ", np.size(y_pred))
    
    
def test_one():
    # Funkcija kojom se predvidja klasa jedne slike (cija se putanja ucitava sa standardnog ulaza)
    
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
    
    try:
        with open("sign_names.json", "r") as f:
            signs = json.load(f)
    except IOError:
        print("Opening file sign_names.json failed!")
        sys.exit()
        
    # U json fajlu trazi naziv znaka predvidjene klase
    # TODO: dodati nazive za sve znakove u json fajlu
    for sign in signs:
        if(sign["class"] == prediction[0]):
            print("Predicted class: {0} (class {1})".format(sign["sign_name"], prediction[0]))
    
    
    #print("Predicted class: ", prediction[0])
    
    

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