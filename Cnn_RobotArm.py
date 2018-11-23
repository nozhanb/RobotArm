# Data preprocessing: Did manually by choosing pics of dogs and cats and splitting
# them properly between training set and test set and cats and dogs folders

# importiong the Kears libraries and packages
from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

##############################################################################
############################ Part 1: BUILDING CNN ##########################
###############################################################################
# Initialising CNN
classifier = Sequential()

# Step 1 - Convolution layer
# Create the feature detector (ex nb of column (pixel) = 3 and row (pixel) of
#  the feature = 3 detector)
# nb_filter = Number of feature map (ex = 32)
# input_shape =(3, 64, 64) to force all the image input to be in the same format
# Colored image are converted to 3D array
# Black and white images are converted to 2D array
# so input_shape = 3 for 3D array and 64 for size
# ATTENTION dans tensorflow input_shape argument are not in the same order
# Here activation function served to remove black pixel to avoid linearity 
# in the model !!! Make sure tensorflow is used in the backend
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation = 'relu'))

# Step 2 - Pooling
# pool_size = max pooling step
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flatenning
classifier.add(Flatten())

# Step 4 - Full connection
# Fully connected means that we create an hidden layer to connect the input layer
# to our ouput layer in the classic NN
# 128 because we have a lot of input nodes. Just based on experience
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



##############################################################################
################### Part 2: FITTING the CNN to the images #####################
###############################################################################
# Go to and copy paste code Processing >ImageDataGenerator https://keras.io/
# Image augmentation to help us to avoir overfitting. Augment the number
# of image we have for the training set
# shear pixel are moved to change a bit the new image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        # rescale allow for all the pixel to range between 0 and 1
        rescale=1./255,
        # Transformation of the immage to not find the same image in the set
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Rescale the image of the test set
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/anhaug/Documents/Robot arm/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        color_mode = 'grayscale')

test_set = test_datagen.flow_from_directory(
                                            'C:/Users/anhaug/Documents/Robot arm/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary',
                                            color_mode = 'grayscale')



# Fit the model
classifier.fit_generator(training_set,
                            # number of images in our training  set
                            steps_per_epoch=2200,
                            epochs=5,
                            validation_data=test_set,
                             # number of images in our test  set
                            validation_steps=800)


import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image

classifier = load_model('RobotArmCNN.h5') 

test_image = image.load_img('C:/Users/anhaug/Documents/Robot arm/dataset/single_prediction/airplane3.jpg', target_size = (64, 64), color_mode = 'grayscale')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 0:
    prediction = 'airplane'
else: 
    prediction = 'apple'
    
    
 