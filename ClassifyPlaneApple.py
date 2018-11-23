# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:12:00 2018

@author: anhaug
"""

import numpy as np
import cv2

from keras.preprocessing import image
from keras.models import Sequential, load_model
from PIL import Image


### Loading the previously built model ###

classifier = load_model('RobotArmCNN.h5') 

test_image = image.load_img('C:/Users/anhaug/Documents/Robot arm/dataset/single_prediction/airplane20.jpg', target_size = (64, 64), color_mode = 'grayscale')

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)


### Check whether the loaded picture is an airplane or an apple ###

if result[0][0] == 0:
    templ = cv2.imread('C:\\Users\\anhaug\\Documents\\Robot arm\\AR_Shared\\airplane_gray.png', 0)
    print("Airplane!")
else: 
    templ = cv2.imread('C:\\Users\\anhaug\\Documents\\Robot arm\\AR_Shared\\apple_gray.png', 0)
    print("Apple!")
    

### Do webcamera stuff/classification ###
    
cap = cv2.VideoCapture(0)
w, h = templ.shape[::-1]


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame, templ,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(frame, top_left, bottom_right, 255, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()