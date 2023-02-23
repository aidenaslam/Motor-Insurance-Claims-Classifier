# Motor-Insurance-Claims-Classifier

CNNs are primarily used for image classification and recognition. The applications in industry are countless - from facial recognition and medical imaging 
to autonomous driving. For my latest project, which I thought of during my time at IQUW (a motor insurance company), I wanted to apply computer vision 
to help automate an actual problem motor insurers face - classification of vehicle damage. Using the Microsoft Bing Image Search REST API in Python, 
I scraped over 1000 images relating to different types of vehicle damage:
  * Dent
  * Scratch
  * Broken light (front or back)
  * Broken windscreen (front or rear)
  * Significant damage
  * No apparent damage
  
 ## Data Processing
 
 After the collection of the data, I pre-processed each image by converting to an image of size 224x224 and then normalising the image by dividing by 255 
 to ensure that each pixel has a similar data distribution. Due to a lack of images, I intentionally did not split the data into training and testing
 
 ## Modelling
 
 There exist many types of CNN architectures and because of the limitations of my computer and lack of images I decided to use Transfer Learning in TensforFlow. 
 The type of CNN considered was an Xception CNN pre-trained on ImageNet. To carry out transfer learning the following steps were used:
  * Exclude the fully connected layer at the top of the network
  * Freeze the model layers so no information is destroyed when training the model using the images scraped
  * New trainable layers added on top of the frozen layer in order to be able to classify vehicle damage. In this case a global average pooling layer and
  dense layer with a softmax activation were used with the number of units set to 6 to represent the 6 different outputs possible for vehicle damage.
  * The final step was to train the model using my dataset of over 1000 images relating to vehicle damage.
  
  The following hyperparameters were chosen:
  * learning rate = 0.2
  * momentum = 0.9
  * decay = 0.01
  * epochs = 5
  * loss = crossentropy
  * optimiser = gradient descent with momentum
    
  The figure below shows the training history of the Xception CNN model constructed. The x-axis displays the number of epochs. Over the 5 epochs, the training accuracy 
  increases from 0.55 to 0.90 and the cross entropyloss decreases to approximately 0.3. This demostrates how powerful CNN's are in image recognition!!
  
  <img src="https://github.com/aidenaslam/Motor-Insurance-Claims-Classifier/blob/dev_aiden/model_training.png" width="600" height="400" />
  
  ## Deployment
  
  * Using Streamlit, a web app was produced and deployed to the streamlit cloud where a user can upload an image and the CNN classifier model constructed would be able to classify the type of damage sustained to the vehicle. Below is a screenshot showing an image of a car that has sustained damage with the output from the CNN classifier model. The model correctly classifies the damage as a dent with probability 0.976.
  
  Link to app: https://aidenaslam-motor-insurance--04-deployment-streamlit-app-l74nsj.streamlit.app/
    
  <img src="https://github.com/aidenaslam/Motor-Insurance-Claims-Classifier/blob/dev_aiden/Streamlit_app_screenshot.png" width="800" height="500" />
  
  
 
 
