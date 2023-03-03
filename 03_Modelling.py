import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from parameters import (preprocessed_images, class_0, class_1, class_2, 
                        class_3, class_4, class_5, model_dir)
from functions import process_image_data


# Load image data for each class
data_class_0_reshaped = process_image_data(preprocessed_images, class_0)
data_class_1_reshaped = process_image_data(preprocessed_images, class_1)
data_class_2_reshaped = process_image_data(preprocessed_images, class_2)
data_class_3_reshaped = process_image_data(preprocessed_images, class_3)
data_class_4_reshaped = process_image_data(preprocessed_images, class_4)
data_class_5_reshaped = process_image_data(preprocessed_images, class_5)

# Create Data Dictionary
data_dict = {
    "data": [data_class_0_reshaped, 
             data_class_1_reshaped, 
             data_class_2_reshaped, 
             data_class_3_reshaped, 
             data_class_4_reshaped, 
             data_class_5_reshaped],
    "label": [np.full((1,data_class_0_reshaped.shape[0]), 0), 
              np.full((1,data_class_1_reshaped.shape[0]), 1), 
              np.full((1,data_class_2_reshaped.shape[0]), 2), 
              np.full((1,data_class_3_reshaped.shape[0]), 3), 
              np.full((1,data_class_4_reshaped.shape[0]), 4), 
              np.full((1,data_class_5_reshaped.shape[0]), 5)]
}

# Combine data
data_combined = tf.concat([data_class_0_reshaped, data_class_1_reshaped], axis=0)
data_combined = tf.concat([data_combined, data_class_2_reshaped], axis=0)
data_combined = tf.concat([data_combined, data_class_3_reshaped], axis=0)
data_combined = tf.concat([data_combined, data_class_4_reshaped], axis=0)
data_combined = tf.concat([data_combined, data_class_5_reshaped], axis=0)

# Combine labels
data_labels_combined = np.concatenate((data_dict['label'][0], data_dict['label'][1]), axis = 1)
data_labels_combined = np.concatenate((data_labels_combined, data_dict['label'][2]), axis = 1)
data_labels_combined = np.concatenate((data_labels_combined, data_dict['label'][3]), axis = 1)
data_labels_combined = np.concatenate((data_labels_combined, data_dict['label'][4]), axis = 1)
data_labels_combined = np.concatenate((data_labels_combined, data_dict['label'][5]), axis = 1)

# Free memory
del data_class_0_reshaped, data_class_1_reshaped, data_class_2_reshaped, data_class_3_reshaped, data_class_4_reshaped, data_class_5_reshaped

# Normalise images
data_combined_normalised = data_combined / 255

# Transfer Learning
n_classes = 6
base_model = tf.keras.applications.xception.Xception(weights = "imagenet",
                                                    include_top= False) # whether to include the fully-connected layer at the top of the network.
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)

model = tf.keras.Model(inputs=base_model.input, outputs= output)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

optimiser = tf.keras.optimizers.legacy.SGD(learning_rate=0.2, momentum=0.9, decay= 0.01)
model.compile(loss="SparseCategoricalCrossentropy", optimizer = optimiser, metrics =["accuracy"])

history = model.fit(data_combined_normalised,
                    data_labels_combined.reshape(1569,1),
                    epochs = 5)

# Save training history
pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
plt.gca().set_ylim(0,3)
plt.savefig('model_training.png')

# Save model
model.save(f'{model_dir}\cnn_model.h5')


