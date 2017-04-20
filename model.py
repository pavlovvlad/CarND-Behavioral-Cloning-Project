import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

### helper functions
# function implements the data augmentation
def data_augmentation(images, angles):
    aug_images, aug_angles = [], []
        
    # flipping
    for image, angle in zip(images, angles):
        aug_images.append(image)
        aug_angles.append(angle)
        aug_images.append(cv2.flip(image,1))
        aug_angles.append(angle*-1.0)

    return aug_images, aug_angles

# function with using of the generator in sense of python 
def generator(samples, batch_size=128):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('\\')[-1]
                path = './data_own/IMG/' + name
                center_image = cv2.imread(path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # save the input data as numpy-arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            #yield images, angles

### constants 
# define sets for trainig and for validation
TRAIN_RATIO = 0.2
BATCH_SIZE = 128 # 32 #64 
EPOCHS = 18
# rows pixels from the top of the image
TOP_RAWS = 74
# rows pixels from the bottom of the image
BOTTOM_RAWS = 20
# columns of pixels from the left of the image
LEFT_COLS = 0
# columns of pixels from the right of the image
RIGHT_COLS = 0

# input image format
ch, row, col = 3, 160, 320


### main run routine
samples = []
with open('./data_own/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# split sets for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=TRAIN_RATIO)
print("Number of samples for training: ", len(train_samples))
print("Number of samples for validation: ", len(validation_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# define the architecture of the model
model = Sequential()
# define the architecture of the model based on NVIDIA CNN-Architecture
# pre-process incoming data, trimming of the image 
model.add(Cropping2D(cropping=((TOP_RAWS,BOTTOM_RAWS),(LEFT_COLS,RIGHT_COLS)), input_shape=(row, col, ch)))
# pre-process incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: ( x / 255.0 ) - 0.5))


# Layer Conv1, input shape: 3x66x320
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))

# Layer Conv2, input shape: 24x31x158 
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))

# Layer Conv3, input shape: 36x14x77
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))

# Layer Conv4, input shape: 48x5x36
model.add(Convolution2D(64, 3, 3))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))

# Layer Conv5, input shape: 64x3x34
model.add(Convolution2D(64, 3, 3))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
    
# Layer Conv6, input shape: 64x1x32
model.add(Flatten())
# Layer Fully connected 7, input shape: 2048
model.add(Dense(160))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Layer Fully connected 8, input shape: 160
model.add(Dense(80))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Layer Fully connected 9, input shape: 80
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Layer Fully connected 10, input shape: 16
model.add(Dense(1))

# visualize the model
from keras.utils.visualize_util import plot
plot(model, to_file='model.png',show_shapes = True)
print("Model plotted")

# train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')   
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# save the trained model
model.save('model.h5')