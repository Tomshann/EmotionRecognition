import os

import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout,MaxPool2D
from keras.utils import img_to_array, load_img
import cv2
import keras

#load model with trained weights
model_file = './models/emotion.keras'
model = Sequential()
model.add(Conv2D(input_shape=(48,48,1),filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096,activation="relu"))
model.add(Dense(7,activation="softmax"))

model.load_weights(model_file)

print(model.summary())

# load categories
batch_size = 128
train_path = 'C:/Users/tomsh/PycharmProjects/Emotion Recognition/archive/train'
categories = os.listdir(train_path)
categories.sort()
num_of_classes = len(categories)

# finding the face in the image using haarcascade


def findFace(path_for_image):
    image = cv2.imread(path_for_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haarCascadeFile = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(haarCascadeFile)
    faces = faceCascade.detectMultiScale(gray)

    for(x,y,w,h) in faces:
        #cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h , x:x+w]

    return roi_gray


def prepareImage(face_img):
    resized = cv2.resize(face_img, (48,48), interpolation=cv2.INTER_AREA)
    img_result = np.expand_dims(resized,axis=0)
    img_result = img_result.astype('float32')  # Convert to float32
    img_result /= 255.0
    return img_result


# test image
test_path = "./andy.jpg"
face_gray_image = findFace(test_path)

img_for_model = prepareImage(face_gray_image)

# run prediction
resultArray = model.predict(img_for_model,verbose=1)
answer = np.argmax(resultArray,axis=1)

text = categories[answer[0]]

img = cv2.imread(test_path)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text,(0,20),font,0.5,(200,20,80),2)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

