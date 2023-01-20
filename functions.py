import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Import an image, convert to np array and resize
def import_image(directory, image, load=True):
    # load the image
    img = load_img(os.path.join(directory, image))
          
    # convert to numpy array
    data = img_to_array(img)
    
    # convert image to size 224 x 224 (for ResNet)
    img_resized = tf.image.resize(data,[224,224])
    return img_resized