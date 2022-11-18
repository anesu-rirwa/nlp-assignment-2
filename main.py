# Rirwa Anesu
# R204432D
# HAI - NLP Assignment 2

import keras
import numpy as np
import os
import tempfile

from opencv import cv2
from os import listdir
from os.path import exists
from keras.applications.inception_v3 import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


import streamlit as st
from PIL import Image

# Download InceptionV3 model
model = InceptionV3()


# Create Streamlit Web App

st.title("Object Classifier Application")

uploaded_video = st.file_uploader("Upload a video", type=['mp4'])
if uploaded_video is not None:
    file_details = {"FileName":uploaded_video.name, "FileType":uploaded_video.type}


submit = st.button(label="Submit", help="Click to classify objects in video")
if submit:
    with open(os.path.join("uploads", uploaded_video.name), "wb") as file:
        file.write(uploaded_video.getbuffer())

        # Split Video into frames

        video_path = 'uploads\\' + file_details['FileName']
        video_capture = cv2.VideoCapture(video_path)

        frame_num = 0

        while (True):
            successFrame, image = video_capture.read()

            if successFrame:
                if frame_num < 21:
                    cv2.imwrite("./images/frame%d.jpg" % frame_num, image)
                else:
                    break

            frame_num += 1
            
        video_capture.release()
        st.success("Video saved and split into frames! Now being fed into model.")

        
pred_list = []
pred_image = []

# Feed the frames into inceptionv3 Model and get objects classified
folder_path = "./images"

img = []

for images in os.listdir(folder_path):
        img.append(images)


for img_path in img:
    frame_jpg = './images/' + img_path
    image = load_img(frame_jpg, target_size=(299, 299))
    image = img_to_array(image)

    # reshape image to inceptionV3 dimensions so we can pass it through the network
    image = np.expand_dims(image, axis=0)
    image = image.reshape((-1, 299, 299, 3))
    image = preprocess_input(image)

    # classify the image
    preds = model.predict(image)
    P = imagenet_utils.decode_predictions(preds, top=5)
    
    # loop over the predictions and display 5 predictions and probability percentage
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        classification = "{}".format(label, i + 1, prob * 100)
        result = classification.replace("_", " ")
        
        if result not in pred_list:
            pred_list.append(result)
            pred_image.append(img_path)
        else:
            continue

# Search Function

search = st.text_input(" Search for objects below..")
search_btn = st.button(label="Search")
if search_btn:
    if uploaded_video is not None:
        if search in pred_list:
            index = pred_list.index(search)
            st.write(f"{search} is in the video objects classified by our model.")

            plot_img = Image.open('images\\' + pred_image[index])

            st.image(plot_img, width=299)

        else:
            st.error(f"Error! {search} is not in the predicted results by our model" )
            st.write('Predicted Objects in the Video')
            for obj in pred_list:
                st.write(obj)
    else:
        st.error("Upload Video first!")