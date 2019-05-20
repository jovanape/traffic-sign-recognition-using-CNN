import os
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 32  # 32x32 images

train_data_dir = "./dataset/train1"
test_data_dir = "./dataset/test1"


def load_data(data_dir):
    # Funkcija kojom se ucitavaju slike iz prosledjenog direktorijuma.
    # Vraca niz slika i niz u kom se nalaze nazivi klasa koje odgovaraju ucitanim slikama.
    
    classes = []
    images = []
    
    data_dir_abs_path = os.path.abspath(data_dir)
    #print(data_dir_abs_path)
    
    file_names = [f for f in os.listdir(data_dir)]
    for f in file_names:
        images.append(skimage.data.imread(os.path.join(data_dir_abs_path, f)))
        classes.append(int(f[:3]))  # prva tri karaktera u nazivu slike oznacavaju klasu
    
    return images, classes


def display_images_and_classes(images, labels):
    # Prikazuje prvu sliku iz svake od klasa.
    # Tako mozemo da vidimo koje sve znakove model moze da nauci
        
    unique_labels = set(labels)
    plt.figure("Existing classes", figsize=(20, 20))
    i = 1
    for label in unique_labels:
        # Za svaku klasu uzima prvu sliku iz images koja joj pripada.
        # index vraca prvo pojavljivanje date klase u listi labels u kojoj se nalaze sve klase sa ponavljanjem
        image = images[labels.index(label)]
        plt.subplot(2, 5, i)  # Slike prikazuje u vidu mreze od 8 redova x 8 kolona
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.axis('off')
        # naziv klase (broj slika u toj klasi)
        plt.title("Class {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.suptitle('Classes of traffic signs in dataset (and number of signs for each class)')
    plt.show()
