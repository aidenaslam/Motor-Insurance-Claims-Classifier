import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf


def preprocess_image(directory, image, load=True):
    """ Import an image, convert to np array and resize """
    # load the image
    img = load_img(os.path.join(directory, image))
          
    # convert to numpy array
    data = img_to_array(img)
    
    # convert image to size 224 x 224 (for ResNet)
    img_resized = tf.image.resize(data,[224,224])
    return img_resized

def process_image_data(dir, filename):
    """ Loads and reshapes images from preprocessed directory"""

    data_dir = os.path.join(dir, filename)
    data = np.load(data_dir, allow_pickle=True)
    data_reshaped = tf.image.resize(data,[224,224])
    return data_reshaped