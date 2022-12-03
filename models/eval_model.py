
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras
import re
import cv2
import tensorflow as tf
from keras_vggface import utils

# Running Configurations
PREDICT = True
SHOW_IMG = True
EVALUATE = True

# Importing Model
model = keras.models.load_model('f-r-i-e-n-d-s.h5')
# preparing data inputs
folders = [x for x in os.listdir(os.getcwd()) if os.path.isdir(x)]
names = [x for x in folders if x in ['chandler',
                                     'joey', 'monika', 'phoebe', 'rachel', 'ross']]

# predicting, showing & Evaluating using the eval_set
eval_x_list = []
eval_y_list = []
folder = os.path.join(os.getcwd(), 'eval_set')
for file in os.listdir(folder):
    file_path = os.path.normpath(os.path.join(folder, file))
    # reading in cv2 format for showing
    if SHOW_IMG:
        inp_img = cv2.imread(file_path)
    # reading in PIL format for prediction & evaluation
    img = image.load_img(
        file_path,
        target_size=(224, 224, 3))
    x = image.img_to_array(img)

    # Appending the data for evaluation at the end
    if EVALUATE:
        eval_x_list.append(x)
        eval_y_list.append(names.index(re.split('_', file)[0]))
    # Predicting to print the outputs
    if PREDICT:
        x = np.expand_dims(x, axis=0)
        results = model.predict(x)
        index = np.argmax(results, axis=1)[0]
        print("-->", file_path, x.shape)
        actual = names[index]
        print(f"actual : {actual}")
        expected = re.split('_', file)[0]
        print(f"expected : {expected}")
        # Showing the images in cv2 native format
        if SHOW_IMG:
            cv2.imshow(actual, inp_img)
            cv2.waitKey(0)

# Final Evaluation
if EVALUATE:
    print(eval_y_list)
    results = model.evaluate(np.asarray(eval_x_list),
                             np.asarray(eval_y_list))
    print("loss : ", results[0], "acc : ", results[1])
