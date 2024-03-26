import keras
import numpy as np
import tensorflow as tf
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
"""
This model was trained using the FER-2013 dataset
Notes: 
Test model using constrative loss function
try more epochs with lower batch sizes
recommended is 128 batch with 250 epoch
"""

# defining the paths to the dataset

train_path = 'C:/Users/tomsh/PycharmProjects/Emotion Recognition/archive/train'
test_path = 'C:/Users/tomsh/PycharmProjects/Emotion Recognition/archive/test'

folder_list = os.listdir(train_path)
folder_list.sort()

X_train = []
Y_train = []

Y_test = []
X_test = []

# load training data into the arrays
for i, folder in enumerate(folder_list):
    files = os.listdir(train_path+"/"+folder)
    for file in files:
        img = cv2.imread(train_path+"/"+folder+"/{0}".format(file),0)
        X_train.append(img)
        Y_train.append(i)  # each folder will be represented by a number


# load testing data into the arrays
for j, folder in enumerate(folder_list):
    files = os.listdir(test_path+"/"+folder)
    for file in files:
        img = cv2.imread(test_path+"/"+folder+"/{0}".format(file),0)
        X_test.append(img)
        Y_test.append(j)  # each folder will be represented by a number


# convert data into a numpy array for the model

X_train = np.array(X_train,'float32')
Y_train = np.array(Y_train,'float32')
X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')


# normalise the data so that values lie between 0 and 1 by performing a division by 255(The max pixel colour)

X_train /= 255.0
X_test /= 255.0

# reshape the images back to their original shape and add a 1 at the end to convert to greyscale

num_of_images = X_train.shape[0]
X_train = X_train.reshape(num_of_images,48,48,1)

num_of_test_images = X_test.shape[0]
X_test = X_test.reshape(num_of_test_images,48,48,1)


# convert labels into categorical format

Y_train = keras.utils.to_categorical(Y_train,num_classes=7)
Y_test = keras.utils.to_categorical(Y_test, num_classes=7)


# building the convolutional neural network model

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(input_shape=input_shape,filters=64, kernel_size=(3,3), padding="same", activation="relu"))
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

print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])

batch = 64
epochs = 10

steps_per_epoch = np.ceil(len(X_train)/batch)
validation_steps = np.ceil(len(X_test)/batch)

stop_early = EarlyStopping(monitor="val_accuracy", patience=5)

# train model

history = model.fit(X_train,Y_train,batch_size=batch,epochs=epochs,verbose=1,validation_data=(X_test,Y_test),shuffle=True,callbacks=[stop_early])

# show the results based on pyplot

acc = history.history['accuracy']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
# show the train and validation accuracy charts

plt.plot(epochs, acc , 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()

# show the train and validation accuracy charts

plt.plot(epochs, loss , 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Training and validation accuracy')
plt.legend(loc='upper right')
plt.show()

#save the model
modelFileName = "./temp/emotion.h5"