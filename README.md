# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The image dataset we taken is mnist and the model must denoise the images and show it with better quality and remove the unwanted noises and learn to show the better version of the images.

Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. MNIST is a dataset of black and white handwritten images of size 28x28.Denoising is the process of removing noise. This can be an image, audio, or document.These noisy digits will serve as our input data to our encoder. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image.

![image](https://github.com/MOHAMED-FAREED-22001617/convolutional-denoising-autoencoder/assets/121412904/cb26a9bd-f9b1-459c-aef2-9fc83fe2b66a)


## DESIGN STEPS

### STEP 1:
Download and split the dataset into training and testing datasets

### STEP 2:
Rescale the data as that the training is made easy

### STEP 3:
Create the model for the program , in this experiment we create to networks , one for encoding and one for decoding.


## PROGRAM
```
Developed by: Mohamed Fareed F
Register number: 212222230082
```
### Import required libraries
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
```
### loading, preprocessing and adding some noise to the data
```python
(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape
x_test.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
     
```
### creating model, compile and fitting it
```python
input_img = keras.Input(shape=(28, 28, 1))                                                     #28,28,,1

hidden_layer = layers.Conv2D(32, (3,3),activation='relu',padding='same')(input_img)  #28,28,32
hidden_layer = layers.MaxPooling2D((2,2),padding='same')(hidden_layer)
hidden_layer = layers.Conv2D(16, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.MaxPooling2D((2,2),padding='same')(hidden_layer)
hidden_layer = layers.Conv2D(4, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.MaxPooling2D((2,2),padding='same')(hidden_layer)
hidden_layer = layers.Conv2D(8, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.MaxPooling2D((2,2),padding='same')(hidden_layer)

encoded = layers.MaxPooling2D((2, 2), padding='same')(hidden_layer)


hidden_layer = layers.Conv2D(8, (3,3),activation='relu',padding='same')(encoded)
hidden_layer = layers.UpSampling2D((2,2))(hidden_layer)
hidden_layer = layers.Conv2D(4, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.UpSampling2D((2,2))(hidden_layer)
hidden_layer = layers.Conv2D(16, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.UpSampling2D((2,2))(hidden_layer)
hidden_layer = layers.Conv2D(32, (3,3),activation='relu',padding='same')(hidden_layer)
hidden_layer = layers.UpSampling2D((2,2))(hidden_layer)
hidden_layer = layers.Conv2D(32, (3,3),activation='relu')(hidden_layer)
hidden_layer = layers.UpSampling2D((2,2))(hidden_layer)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid',padding='same')(hidden_layer)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```
### predict the noisy added test data
```python
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show() 
```

## OUTPUT

### parameters for the model

![image](https://github.com/MOHAMED-FAREED-22001617/convolutional-denoising-autoencoder/assets/121412904/e21393a0-3405-44b9-aa2d-fa044599fd38)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/MOHAMED-FAREED-22001617/convolutional-denoising-autoencoder/assets/121412904/386fa1c7-d4dd-44c1-913a-e8e14101aedb)





## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
