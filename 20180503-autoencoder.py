
## 2018.5.5
## Convolutional autoencoder
## Works fine thought the acc is low 'cause epochs=50.
## https://blog.keras.io/building-autoencoders-in-keras.html

from mnist import MNIST
mndata = MNIST('samples')

import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()  #Discard the labels

(x_train, _) = mndata.load_training()
x_train = np.asarray(x_train)
x_train = x_train[: 100, : 784]

(x_test, _) =mndata.load_testing()
x_test = np.asarray(x_test)
x_test = x_test[: 10, : 784]

# Normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test),28,28,1))

print(x_train.shape)
print(x_test.shape)

import sys
#sys.exit(0)

# Train autoencoder
#autoencoder.fit(x_train, x_train,
#                epochs=50,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_test, x_test))


#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

#-----------------------------------------------------------------
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)  #(28,28,16)
x = MaxPooling2D((2,2), padding='same')(x)  #(14,14,16), n=28,f=2,p=0,s=f
x = Conv2D( 8,(3,3), activation='relu', padding='same')(x)  #(14,14,8)
x = MaxPooling2D((2,2),padding='same')(x)   #(7,7,8)
x = Conv2D( 8,(2,2), activation='relu', padding='same')(x)  #(7,7,8)
encoded = MaxPooling2D((2,2), padding='same')(x)    #(4,4,8)

# at this point the representation is (4,4,8) 

x = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)     #(4,4,8)
x = UpSampling2D((2,2))(x)  #(8,8,8)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)   #(8,8,8)
x = UpSampling2D((2,2))(x)  #(16,16,8)
x = Conv2D(16,(3,3), activation='relu')(x)  #(14,14,16)
x = UpSampling2D((2,2))(x)  #(28,28,16)
decoded = Conv2D(1,(3,3), activation='sigmoid', padding='same')(x)  #(28,28,1)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


autoencoder.summary()

import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show() 


