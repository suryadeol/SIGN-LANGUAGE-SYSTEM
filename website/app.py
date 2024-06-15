import sys
import os
import glob
import re
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from PIL import Image
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from time import sleep, time

import tensorflow as tf
from tensorflow.keras import models, layers
import math
import matplotlib

# Use the 'agg' backend for Matplotlib,  whihc is neccesarily need for matplotlib module in flask. 
matplotlib.use('agg')
import matplotlib.pyplot as plt


from matplotlib.image import imread
import cv2
import csv
from PIL import Image

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)


################################### Only for these project #########################

import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from keras.models import load_model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

################################ All pages server connection/routings ##########################################

@app.route("/", methods=["GET", "POST"]) 
def runhome():
    return render_template("index.html") 

@app.route("/server_home", methods=["GET", "POST"]) 
def server_home():
    return render_template("index.html") 

@app.route("/about", methods=["GET", "POST"]) 
def about():
    return render_template("about.html") 

@app.route("/testimonial", methods=["GET", "POST"]) 
def testimonial():
    return render_template("testimonial.html") 

@app.route("/404", methods=["GET", "POST"]) 
def _404():
    return render_template("404.html") 

@app.route("/contact", methods=["GET", "POST"]) 
def contact():
    return render_template("contact.html") 

@app.route("/sign_language_prediction", methods=["GET", "POST"]) 
def sign_language_prediction():
    return render_template("sign_language_prediction.html") 

@app.route("/sign_language_recognition", methods=["GET", "POST"]) 
def sign_language_recognition():
    return render_template("sign_language_recognition.html")     

############################---   stress analysis ---########################################

# Load the trained model
model_path = './model/sign_language_MAIN_1.h5'
model = load_model(model_path)
class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
                 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

@app.route('/gesture_predict', methods=['GET', 'POST'])
def gesture_predict():
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Process image (input) for the model
        def process(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            img = tf.keras.applications.mobilenet.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            return img
    
        # Process the uploaded image for prediction
        processed_image = process(cv2.imread(file_path))

        # Get the prediction from the model
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_mapping.get(predicted_class_index, 'Unknown')
        maxconfidence = np.max(prediction)

        # Render the result template
        return render_template("sign_language_prediction_result.html", predicted_class=predicted_class, maxconfidence=maxconfidence)

################################################################################################

@app.route('/gesture_recognition', methods=['GET', 'POST'])
def gesture_recognition():
    if request.method == 'POST':


        # Load the pre-trained model
        model = load_model('./model/sign_language_main_1.h5')

        # Map class indices to letters
        class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
                         10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
                         19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

        # Open the camera
        cap = cv2.VideoCapture(0)

        # Initialize variables for storing and displaying letters and words
        detected_letters = []
        detected_word = ""
        start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            # Increase the camera size
            frame = cv2.resize(frame, (800, 600))

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply hand detection logic here (use your hand detection code)
            # For demonstration purposes, let's assume hand_roi is the region of interest containing the hand
            # Calculate the center of the frame
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            rect_size = 250  # Size of the hand detection rectangle
            top_left = (frame_center[0] - rect_size // 2, frame_center[1] - rect_size // 2)
            bottom_right = (top_left[0] + rect_size, top_left[1] + rect_size)

            # Resize and preprocess the hand ROI
            hand_roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            processed_hand = cv2.resize(hand_roi, (224, 224))
            processed_hand = np.expand_dims(processed_hand, axis=0)
            processed_hand = processed_hand / 255.0  # Normalize the pixel values

            # Get the prediction from the model
            prediction = model.predict(processed_hand)
            predicted_class = np.argmax(prediction)

            # Get the predicted sign based on the class mapping
            predicted_sign = class_mapping.get(predicted_class, 'Unknown')

            # Draw a rectangle around the hand
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Hand detection rectangle at the center

            # Display the predicted sign with background color
            sign_bg_color = (255, 0, 0)  # Specify the background color
            sign_text_color = (255, 255, 255)  # Specify the text color
            cv2.rectangle(frame, (10, 10), (200, 50), sign_bg_color, -1)
            cv2.putText(frame, predicted_sign, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, sign_text_color, 2, cv2.LINE_AA)

            # Check if the detected sign is 'nothing' (hand not detected)
            if predicted_sign == 'nothing':
                # Reset the timer when hand is not detected
                start_time = None
            else:
                # If hand is detected, start or update the timer
                if start_time is None:
                    start_time = time.time()
                else:
                    # Check if the detected sign is displayed for more than 3 seconds
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 3:
                        if predicted_sign == 'space':
                            detected_letters.append('_')  # Add underscore when SPACE is detected
                        elif predicted_sign == 'del':
                            if detected_letters:
                                detected_letters.pop()  # Remove the last letter from the list
                            detected_word = ''.join(detected_letters)  # Update the detected word
                        else:
                            detected_letters.append(predicted_sign)
                        detected_word = ''.join(detected_letters)
                        start_time = None  # Reset the timer

            # Draw a rectangle around the detected word
            word_bg_color = (0, 0, 255)  # Specify the background color
            word_text_color = (255, 255, 255)  # Specify the text color
            cv2.rectangle(frame, (10, 60), (500, 90), word_bg_color, -1)
            cv2.putText(frame, f'Detected Word: {detected_word}', (20, 80 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, word_text_color, 2, cv2.LINE_AA)

            # Display the frame with the hand detection and prediction
            cv2.imshow('Hand Detection and Prediction', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


        # Directly send all confidences
        return render_template("sign_language_recognition.html")


################################################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
