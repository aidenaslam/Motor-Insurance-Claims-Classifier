# This script pre-processes the scraped images

# import libraries
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import tensorflow as tf

from parameters import saved_images
from functions import import_image

# Create a list of the classes
class_list = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5"]

# Create dictionary with classes as keys
images_in_folder = {}
for c in class_list:
    images_in_folder[c] = None

# Fill in values in dictionary with the names of each image
for c in class_list:
    image_dir = os.path.join(saved_images, c)
    images_in_folder[c] = [image for image in listdir(image_dir) if isfile(join(image_dir, image))]

# Create another dictionary to store the data of each image
images_for_each_class = {}
for c in class_list:
    images_for_each_class[c] = None

# Save images for each class in dictionary
for c in images_in_folder.keys():
    image_dir = os.path.join(saved_images, c)
    images_for_each_class[c] = np.array([np.array(import_image(image_dir, image)) for image in images_in_folder[c]])

# Save pre-processed data for modelling
for c in images_in_folder.keys():
    images_for_each_class[c].dump(f"data_{c}.npy")