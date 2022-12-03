
# preparing data inputs
import os
from tensorflow.keras.preprocessing import image
from keras_vggface import utils
import numpy as np

# Building Model
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense
import tensorflow.keras as keras
import tensorflow as tf
import re

# preparing data inputs
folders = [x for x in os.listdir(os.getcwd()) if os.path.isdir(x)]
chosen = [x for x in folders if x in ['chandler',
                                      'joey', 'monika', 'phoebe', 'rachel', 'ross']]
# Preparing training data
X_LIST = []
Y_LIST = []
NAME_LIST = []
for i in range(len(chosen)):
    print(chosen[i])
    for root, _, files in os.walk(chosen[i]):
        train_list = []
        for file in files[0:50]:
            print("->", file)
            # load the image
            img = image.load_img(
                os.path.join(chosen[i], file),
                target_size=(224, 224, 3))
            # prepare the image

            x = image.img_to_array(img)
            X_LIST.append(x)
            Y_LIST.append(i)
            NAME_LIST.append(chosen[i])

# print(NAME_LIST)
# print(Y_LIST)
# print(X_LIST)

# Building Model
base_model = VGGFace(include_top=False,
                     model='senet50',
                     input_shape=(224, 224, 3))

NO_CLASSES = len(chosen)

# Defining Trainable layers
flat_layer = Flatten(name='flatten')
# final layer with softmax activation
out_layer = Dense(NO_CLASSES, activation='softmax', name='classifier')

# Stacking layers
model = keras.Sequential([
    base_model,
    flat_layer,
    out_layer
])
# Freeze the first layer
base_model.trainable = False
# model.summary()

# Printing Layers Information
for layer in model.layers:
    print(layer.trainable)

# Compiling Model with sparse_categorical_crossentropy
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
X_LIST = np.asarray(X_LIST)
Y_LIST = np.asarray(Y_LIST)
# print(X_LIST.shape, X_LIST.dtype)
# print(Y_LIST.shape, Y_LIST.dtype)
model.fit(x=X_LIST, y=Y_LIST,
          #   batch_size=1,
          #   verbose=1,
          epochs=20)

# summary
model.summary()

# Preparing Evaluation Dataset
eval_x_list = []
eval_y_list = []
folder = os.path.join(os.getcwd(), 'eval_set')
for file in os.listdir(folder):
    file_path = os.path.normpath(os.path.join(folder, file))
    # print(file_path)
    img = image.load_img(
        file_path,
        target_size=(224, 224, 3))
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    eval_x_list.append(x)
    eval_y_list.append(chosen.index(re.split('_', file)[0]))
    print("-->", file_path, x.shape)

print(eval_y_list)
# evaluating Model
results = model.evaluate(np.asarray(eval_x_list),
                         np.asarray(eval_y_list))
print("loss : ", results[0], "acc : ", results[1])
# Saving Model in keras native format
model.save('f-r-i-e-n-d-s.h5')
model = tf.keras.models.load_model('f-r-i-e-n-d-s.h5')
# Saving Model in Tf-lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("f-r-i-e-n-d-s.tflite", "wb").write(tflite_model)
