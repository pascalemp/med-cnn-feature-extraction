# Main Python Code - Less interested in performance of model, 
# more interested in investigating intermediary layer outputs.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

#create a df containing the locations of each image and their directories
def create_dir_df():
    
    df = pd.DataFrame(columns=['Class','Directory'])
    basedir = './datasets/' 
    category = ['val','test','train']

    for choice in category:
        for folder in os.listdir(basedir+choice+'/'):
            if os.path.isdir(basedir+choice+'/'+folder):
                for Class in os.listdir(basedir+choice+'/'+folder+'/'):
                        df = df.append({'Class':folder,'Directory':basedir+choice+'/'+folder+'/'+Class},ignore_index=True)

    data_sample = df.sample(frac = 1) #Axis = 1, i.e. show a sample of the classes.
    return data_sample

df = create_dir_df()

train_image = []
test_image = []

for location in df.iloc[:]['Directory']:
    img = cv2.imread(location,0)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
    img = img.reshape(64,64,1)
    train_image.append(img)
X = np.array(train_image)

#DATA TO BE USED FOR MODEL TRAINING
y = np.array(df.iloc[:]['Class'])
y = y.reshape(y.shape[0],1)
one_hot_encode = OneHotEncoder(handle_unknown='ignore') #ignore any data items that are invalid.
one_hot_encode.fit(y)
y = one_hot_encode.transform(y).toarray() #convert to numpy array in order to perform data analysis.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #20% Test Data, 80% Train Data

num_normal_images =  df[df['Class']== 'NORMAL'].count()[0]
num_pneu_images =  df[df['Class']== 'PNEUMONIA'].count()[0]

print('NORMAL Images: {}'.format(num_normal_images))
print('PNEUMONIA Images: {}'.format(num_pneu_images))
print('{} times more images of the PNEUMONIA class than NORMAL class.'.format((num_pneu_images/num_normal_images).round(2)))

# Dataset balancing

total_train_images = X_train.shape[0]

imbalance_factor = np.log([num_pneu_images/num_normal_images])

norm_weight = (1 / num_normal_images)*(total_train_images)/2 #Inverse of amount of images * 1/2 total images 
pneu_weight = (1 / num_pneu_images)*(total_train_images)/2

class_weights = {0: norm_weight, 1: pneu_weight}

print('Weight for NORMAL class: {:.2f}'.format(norm_weight))
print('Weight for PNEUMONIA: {:.2f}'.format(pneu_weight))

input_shape = X_train.shape[1:]
output_shape = y_train.shape

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=1))

model.add(tf.keras.layers.Conv2D(32, (1, 1), padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Activation("softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])

history = model.fit(x = X_train, y = y_train, validation_split = 0.3, batch_size = 32, epochs = 10, class_weight = class_weights)

# Plot results graphs 

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Hide any tensorflow memory warning messages.

layer = model.layers 

# Method to show intermediate layer outputs for analysis

def show_layer_output(layer_index):

    #Plot Conv2D Feature Outputs (can be used for any layer) 

    conv_layer_index = [layer_index]  #Index of a Conv2D Layer (In our case CONV2D Layers at 0 and 3... etc)
    outputs = [model.layers[i].output for i in conv_layer_index]

    model_trunc = tf.keras.Model(inputs=model.inputs, outputs=outputs)

    # Generate feature output by predicting on the input image
    feature_output = model_trunc.predict(X_test[layer_index].reshape(1,64,64))

    columns = 4
    rows = 4

    for ftr in feature_output:

        fig=plt.figure(figsize=(20, 20))

        for i in range(1, columns*rows +1):
            fig =plt.subplot(rows, columns, i)
            fig.set_xticks([]) 
            fig.set_yticks([])
            plt.imshow(ftr[:, :, i-1], cmap='gray')

        plt.show()

show_layer_output(11) # Show some layer output, in this case layer 11