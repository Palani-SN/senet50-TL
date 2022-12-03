import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import re

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = [os.path.join(os.getcwd(), 'test_set', x)
               for x in os.listdir(os.path.join(os.getcwd(), 'test_set'))]

model = keras.models.load_model('models/f-r-i-e-n-d-s.h5')

names = ['chandler', 'joey', 'monika', 'phoebe', 'rachel', 'ross']

Width = None
Height = None

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        inp_image = cv2.imread(file)
        Width = inp_image.shape[1]
        Height = inp_image.shape[0]
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(
            cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            continue
        annotated_image = inp_image.copy()
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            act_xmin = int(box.xmin*Width)
            act_ymin = int(box.ymin*Height)
            act_width = int(box.width*Width)
            act_height = int(box.height*Height)
            left = act_xmin
            right = act_xmin+act_width
            top = act_ymin
            bottom = act_ymin+act_height

            cropped = inp_image[top:bottom, left:right]
            converted = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = Image.fromarray(converted)
            resized = cropped.resize((224, 224))
            x = image.img_to_array(resized)
            x = np.expand_dims(x, axis=0)
            results = model.predict(x)
            index = np.argmax(results, axis=1)[0]
            name = names[index]

            cv2.rectangle(annotated_image, (left, top),
                          (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(annotated_image, (left-1, bottom),
                          (right+1, bottom+35), (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(annotated_image, name, (left + 15,
                        bottom+25), font, 1.0, (255, 255, 255), 1)
        cv2.imwrite(
            f"output-{re.split('[.]', os.path.basename(file))[0]}.jpg", annotated_image)
