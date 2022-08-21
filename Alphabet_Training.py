# Importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf

# Choosing the batch and image size
batch_size = 32
img_height = 256
img_width = 256


# Defining training data part. Taking 80% for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_alphabet_train/asl_alphabet_train/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# Defining validation data part. Taking 20% for validation
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "asl_alphabet_train/asl_alphabet_train/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)



class_names = train_ds.class_names
print("Class names:", class_names)
print("Total classes:", len(class_names))



import matplotlib.pyplot as plt

# Showing some sample dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(29):
        ax = plt.subplot(6, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


from tensorflow.keras.layers import Rescaling
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Defining the model architecture.
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)), # Rescaling each pixel value to between 0 - 1 - To rescale the pixel value
    layers.Conv2D(16, 3, padding='same', activation='relu'), # Applying the first convolutional layer with 16 filters and kernel size of 3
    layers.MaxPooling2D(), # Applying MaxPooling - From kernal size of 2 x 2, it takes first 2 maximum pixel value
    layers.Conv2D(32, 3, padding='same', activation='relu'), # Applying the second convolutional layer with 32 filters and kernal size of 3
    layers.MaxPooling2D(),# Applying MaxPooling again
    layers.Conv2D(64, 3, padding='same', activation='relu'), # Applying the third convolutional layer with 64 filters and kernal size of 3
    layers.MaxPooling2D(),# Applying MaxPooling again
    layers.Flatten(), # Reformatting the matrix from convolutional layer
    layers.Dense(128, activation='relu'), # Adding a dense layer to the network
    layers.Dense(29, activation='softmax') # Adding the output layer with softmax activation
])

model.summary()

# Defining the loss and optimizer, Sparse for prediciton and adam for minimizing loss
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model with epochs = 100
model.fit(train_ds, batch_size=32, validation_batch_size=32, validation_data=test_ds, epochs=100)

# Saving the model
model.save("ASL_Project")
