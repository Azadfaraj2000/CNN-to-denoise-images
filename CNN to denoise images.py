# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:21:32 2021

@author: af4g18
"""
import pandas as pd
import numpy as np
import sklearn
import keras 
from matplotlib import pyplot as plt
import keras.datasets.mnist as mnist
import sklearn.preprocessing 
(x_train, y_train), (x_test, y_test) =  mnist.load_data(path="mnist.npz")

# create scaler
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
# this scaler is applied to 2D arrays, so we need to apply them to the data 
# one image at a time

#for processing, it is better to use floating point
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


for i in range(10):  # how many records we will display
    x = x_train[(y_train==i).squeeze(),:,:]
    plt.subplot(2,5,i+1)
    plt.imshow(x[0,:,:])
    plt.axis('off')
plt.plot()


#%% It is important to note the input size

print(x_train.shape)
print(x_test.shape)


# The first dimension is the number of training images, whilst dimensions 
# 2,3,4 are the image coordinates. These are 28 by 28 images

#%% Pre-process the image data

# create scaler
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
# this scaler is applied to 2D arrays, so we need to apply them to the data 
# one image at a time
def im_scaler(x,scaler):
    for i, X in enumerate(x[:,:,:]):
        #In order to fit a neural network framework for model training, 
        #stack all the 28 x 28 = 784 values in a column. 
        x[i,:,:] = scaler.fit_transform(X.reshape(28*28,1)).reshape(28,28)
    return x

X_train = im_scaler(x_train,scaler)
X_test = im_scaler(x_test,scaler)
#The images are 28x28 and 1 color channel 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  #specify colour channel
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) #specify colour channel

#In an auto-encoder, the output data is the same as the input data
Y_train = X_train
Y_test = X_test
#%% The convolutional autoencoder model

#loss of info when data is sliced and stacked
#convolutional autoencoder retains spartial info of image data
c = 30 #size of the code layer

cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3),
                            strides=(2,2), 
                            activation='relu', 
                            input_shape=X_train.shape[1:]))
cnn.add(keras.layers.Conv2D(32, kernel_size=(3,3),  
                            activation='relu'))  
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3),
                            strides=(2,2),  
                            activation='relu'))  
cnn.add(keras.layers.Conv2D(64, kernel_size=(3,3),  
                            activation='relu'))  
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(100, activation='sigmoid'))   
cnn.add(keras.layers.Dense(c, activation='sigmoid'))  
cnn.add(keras.layers.Dense(7*7*32, activation='sigmoid'))    
cnn.add(keras.layers.Reshape((7,7,32)))
cnn.add(keras.layers.Conv2DTranspose(32, (3, 3), 
                                     activation='relu', 
                                     padding='same' ,
                                     strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(64, (3, 3), 
                                     activation='relu', 
                                     padding='same' ,strides=(2, 2)))
cnn.add(keras.layers.Conv2DTranspose(1, (3, 3), 
                                     activation='relu', 
                                     padding='same' ))

cnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
cnn.summary()
#%% Train the autoencoder over 5 epochs (number of iterations)
#note: X_train is used as both our input data and output
history = cnn.fit(X_train, X_train, validation_data = (X_test,X_test),
                    epochs=5, batch_size=64)
                              #number of samples
print(cnn.count_params())

#%%
pred = cnn.predict(X_train[0].reshape(-1,28,28,1))
plt.imshow(pred.reshape((28,28)),cmap='gray')

#%% split the network into two models
def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
  # create an empty model
  new_model = keras.Sequential()
  for ix in range(starting_layer_ix, ending_layer_ix + 1):
    curr_layer = main_model.get_layer(index=ix)
    # copy this layer over to the new model
    new_model.add(curr_layer)
  return new_model

encoder = extract_layers(cnn,0,6)
decoder = extract_layers(cnn,7,11)

code5 = encoder.predict(X_train[0].reshape(-1,28,28,1))
decode5 = decoder.predict(code5)

code0 = encoder.predict(X_train[1].reshape(-1,28,28,1))
decode0 = decoder.predict(code0)
#encoder generates a code for a given image. 
#The decoder than takes the code and recreates the image

#%% Using the code to interpolate between numbers
for i,c in enumerate(np.arange(0,1,0.1)):
    print(c)
    code = c*code0+(1-c)*code5
    decode = decoder.predict(code)
    plt.subplot(2,5,i+1)
    plt.imshow(decode.reshape((28,28)),cmap='gray')
    plt.axis('off')
plt.title("encoder to interpolate between images")
plt.tight_layout()
plt.show()

#interpolating the images directly
for i,c in enumerate(np.arange(0,1,0.1)):
    print(c)
    interp = c*X_train[1].reshape(28,28)+(1-c)*X_train[0].reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(interp,cmap='gray')
    plt.axis('off')
plt.title("interpolating the images directly")
plt.tight_layout()
plt.show()

#%% Using the autoencoder for de-noising generate training data with poisson noise 
X_trainN = np.random.poisson(X_train)
X_testN = np.random.poisson(X_test)

for i in np.arange(5):
    #noise free image
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    #noisy image
    plt.subplot(2,5,i+1+5)
    plt.imshow(X_trainN[i].reshape(28,28),cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

#%% train the model with the noisy data as the inputs, and the clean data the outputs
# learn how to de-noise the images
daa = cnn
daa.fit(X_trainN, X_trainN, validation_data = 
                  (X_testN,X_test), epochs=5, batch_size=64)



#%% Exercise: Predict the output for a few examples of the noisy test samples
#X_testN and show the results, plotting the noise free image, the noisy image
#and the predicted denoised image.
#plot denoised images
pred = daa.predict(X_testN)

for i in np.arange(5):
    #noise free image
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    #noisy image
    #plt.subplot(2,5,i+1+5)
    #plt.imshow(X_testN[i].reshape(28,28),cmap='gray')
    #plt.axis('off')
    
plt.tight_layout()
plt.show()

for i in np.arange(5):
    #predicted denoised image
    plt.subplot(2,5,i+1)
    plt.imshow(X_testN[i].reshape(28,28),cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

for i in np.arange(5):
    #predicted denoised image
    plt.subplot(2,5,i+1)
    plt.imshow(pred[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    plt.axis('off')
plt.tight_layout()
plt.show()

