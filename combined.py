import logging
import threading
from pathlib import Path
from typing import List, NamedTuple
from load_model import model
import os
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import torch
import queue
import shutil
from torch.utils.data import DataLoader, TensorDataset
import glob
import av
import tensorflow as tf
import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import time
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Assigning MediaPipe variables
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Defining a function to load the ASL model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("ASL_Weight")
    return model

# ASL classes
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Loading the ASL model
new_model = load_model()

# Defining a function to detect all the poses using MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

# Defining a function to help drawing the detected segments on frames
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

# Defining a function to extract the features from the frames
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    key_points = np.concatenate([pose, face, lh, rh, np.zeros(99)])
    return key_points.reshape(24, 24, 3)

# Defining a function to add padding to numpy array
def padding_frame(npy_folder):
    all_npy = len(glob.glob(f"{npy_folder}/*npy"))
    for j in range(all_npy, 50):
        keypoints = np.zeros((24, 24, 3))
        npy_path = os.path.join(npy_folder, str(j))
        np.save(npy_path, keypoints)

#Defining a function to prepare the data for training and testing
def prepare_data(seq_length, npy_folder):
    sequence_length = seq_length
    sequences = []
    window = []
    for frame_num in range(sequence_length):
        res = np.load(os.path.join(npy_folder, "{}.npy".format(frame_num)))
        window.append(res)
    sequences.append(window)
    sequences = np.array(sequences).transpose((0, 4, 1, 2, 3))
    return sequences


def main():
    app_object_detection()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Detection(NamedTuple):
    name: str
    prob: float


class Alphabet(NamedTuple):
    Content: str
    name: float

# Word video labels
labels = ['Candy', 'Clothes', 'Computer', 'Cousin', 'Book', 'Before', 'Go', 'Chair', 'Who', 'Drink']
default_folder = "video_extracted"


def app_object_detection():
    option = st.sidebar.radio('Select the Option',
                              ['Alphabet',
                               'Words'], )
    if option == "Alphabet":
        st.title("Sign Language Model for Alphabet Classification")
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) # Use for uploading the image
        print(type(img_file_buffer))
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer)) # Loading the image
            roi = cv2.resize(image, (256, 256)) # Resizing because the model is trained on 256, 256
            test_image = np.expand_dims(roi, axis=0) # To add extra dimensions in image
            result = new_model.predict(test_image) # This is where the prediction is happening
            print(result)
            sm = torch.nn.Softmax()
            probabilities = sm(torch.from_numpy(result)) # Calculating the probability using softmax
            print(probabilities)
            prediction = class_names[np.argmax(result)] # Getting the predicted class
            print(image.shape)
            print(prediction)
            st.image(
                image, caption=f"Processed image"
            )
            labels_placeholder = st.empty()
            result = list()
            result.append(Alphabet(Content="Predicted", name=prediction))
            labels_placeholder.table(result) # Showing the results
        else:
            st.write("Please Upload Image First")
    else: # This is for word classification
        st.title("Sign Language Model for Words Classification")
        if os.path.exists(default_folder): # Creating a folder to store the numpy array's
            try:
                shutil.rmtree(default_folder)
                os.mkdir(default_folder)
            except:
                pass
        ff = st.file_uploader("Please Upload the file")
        image_placeholder = st.empty()
        if ff is not None:
            print("Got the Material")
            base_folder = "videos"
            main_path = os.path.join(base_folder, ff.name)
            if os.path.exists(main_path): # Saving the videos locally - line 181
                os.remove(main_path)
            with open(main_path, "wb") as f:
                f.write(ff.read())
            npy_folder = default_folder
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                frame_count = 0
                vidcap = cv2.VideoCapture(main_path) # Loading the videos
                success = True
                while success:
                    success, frame = vidcap.read()  # get next frame from video
                    if success:
                        image, results = mediapipe_detection(frame, holistic) # MediaPipe Detection
                        keypoints = extract_keypoints(results) # Extracting the key points
                        npy_path = os.path.join(npy_folder, str(frame_count)) # Defining the numpy path
                        np.save(npy_path, keypoints) # Saving the frames to numpy array's
                        frame_count += 1
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        draw_styled_landmarks(image, results) # Shows the detected key points on frames
                        image_placeholder.image(image, caption='Video', width=500)
                        time.sleep(0.09)
                    else:
                        break
            padding_frame(npy_folder) # Apply padding
            sequences = prepare_data(50, default_folder) # Preparing the data for testing
            test_dataloader = DataLoader(TensorDataset(torch.Tensor(sequences)), batch_size=1)
            for inputs in test_dataloader:
                # move inputs and labels to the device the training is taking place on
                inputs = inputs[0]
                outputs = model(inputs) # Doing the prediction
                sm = torch.nn.Softmax()
                probabilities = sm(outputs) # Calculating the probability
                print("Check the probability")
                print(probabilities)
                _, preds = torch.max(probabilities, 1)
                # clean_output = outputs.detach().numpy().tolist()[0]
                clean_output = probabilities.detach().numpy().tolist()[0]
                print(clean_output)
                predicted_index = preds.numpy().tolist()[0]
                labels_placeholder = st.empty()
                result = list()
                result.append(Detection(name=labels[predicted_index], prob=float(clean_output[predicted_index])))
                labels_placeholder.table(result)
            st.write("This is the prediction on upload")


if __name__ == "__main__":
    main()
