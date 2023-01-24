import streamlit as st
import joblib
from PIL import Image
import numpy as np
import time
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
#import image 
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

cnn_model = tf.keras.models.load_model('cnn_model.h5')
#cnn_model = pickle.load(open("cnn_model.pkl", 'rb'))
#cnn_model = joblib.load("cnn_model.pkl")

# Designing the interface
st.title("Motor Accident Classification App")
# For newline
st.write('\n')

# Description of App
st.write('This app allows the user to upload an image of their vehicle if it has sustained any damage, and then classify the type of damage it has sustained.')
st.write('It can currently classify the following types of damage:')
st.write("- Dents")
st.write("- Scratches")
st.write("- Broken lights (front or rear)")
st.write("- Broken windows")
st.write("- Significant damage (for example a write-off)")
st.write('\n')
st.write("Instructions to use the app:")
st.write("1) First upload an image and wait for the image to load on the interface")
st.write("2) Click the classify button and wait for the model to predict the type of damage the car has sustained.")

# Front Page Image
image = Image.open('rio_de_janeiro.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    
    # convert to numpy array
    image = img_to_array(u_img)
    
    # convert image to size 224 x 224 (for ResNet)
    my_image = tf.image.resize(image,[224,224])
    # Normalise image
    my_image = my_image / 255

    # Alter dim for prediction with model
    my_image = tf.expand_dims(my_image, axis=0)

    
# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):      
            
            # Return predictions as ndarray int
            test_proba = cnn_model.predict(my_image)            
            test_proba = test_proba.tolist()
            
            # flatten nested list
            test_proba = [a for b in test_proba for a in b]
            
            # round prediction to identify type of damage
            test_proba_rounded = [round(x) for x in test_proba]
            
            time.sleep(1)
            st.success('Done!')
            st.balloons()
            
        st.sidebar.header("CNN Predicts: ")
        
        ## Classify type of damage
        for idx, x in enumerate(test_proba_rounded):
            if x == 1:
                if idx == 0:
                    st.sidebar.write(f"The car has a dent, with probability {round(test_proba[idx],3)}")
                elif idx == 1:
                    st.sidebar.write(f"The car has scratches, with probability {round(test_proba[idx],3)}")
                elif idx == 2:
                    st.sidebar.write(f"The car has broken lights, with probability {round(test_proba[idx],3)}")
                elif idx == 3:
                    st.sidebar.write(f"The car has window damage, with probability {round(test_proba[idx],3)}")
                elif idx == 4:
                    st.sidebar.write(f"The car has significant damage, with probability {round(test_proba[idx],3)}")
                elif idx == 5:
                    st.sidebar.write(f"The car has no apparent damage, with probability {round(test_proba[idx],3)}")        
        
