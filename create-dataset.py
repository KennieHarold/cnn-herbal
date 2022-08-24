import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'train'
TEST_DIR = 'test'
OUT_DIR = 'out'
IMG_SIZE = 150


def label_img(word_label):
    if word_label == 'lagundi': return [1, 0, 0, 0, 0]
    elif word_label == 'mayana': return [0, 1, 0, 0, 0]
    elif word_label == 'oregano': return [0, 0, 1, 0, 0]
    elif word_label == 'sambong': return [0, 0, 0, 1, 0]
    elif word_label == 'yerba': return [0, 0, 0, 0, 1]
    else: pass

    
def create_train_data():
    training_data = []
    for directory in tqdm(os.listdir(TRAIN_DIR)):
        image_dir = TRAIN_DIR + "/" + directory
        for img in tqdm(os.listdir(image_dir)):
            label = label_img(directory)
            path = os.path.join(image_dir, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        
    shuffle(training_data)
    np.save('out/train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for directory in tqdm(os.listdir(TEST_DIR)):
        image_dir = TEST_DIR + "/" + directory
        for img in tqdm(os.listdir(image_dir)):
            label = label_img(directory)
            path = os.path.join(image_dir, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), np.array(label)])
        
    shuffle(testing_data)
    np.save('out/test_data.npy', testing_data)
    return testing_data


if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


create_train_data()
create_test_data()