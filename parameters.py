import os

# logs dir
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Bing Search V7 subscription key and endpoint
subscription_key = "ff261e80f649468dab61f5f938238407"
endpoint = 'https://api.bing.microsoft.com/v7.0/images/search'

# where to save scraped images
saved_images = os.path.join(os.getcwd(), "01_Image_Scrape", "Data")
if not os.path.exists(saved_images):
    os.makedirs(saved_images) 


# where to save preprocessed images
preprocessed_images = os.path.join(os.getcwd(), "02_Processed_Images")
if not os.path.exists(preprocessed_images):
    os.makedirs(preprocessed_images) 

# preprocessed file_names 
class_0 = 'data_class_0.npy'
class_1 = 'data_class_1.npy'
class_2 = 'data_class_2.npy'
class_3 = 'data_class_3.npy'
class_4 = 'data_class_4.npy'
class_5 = 'data_class_5.npy'

# model directory
model_dir = os.path.join(os.getcwd(), "03_Saved_Models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir) 