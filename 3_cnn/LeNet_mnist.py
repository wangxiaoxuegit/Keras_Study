
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D, Conv2D
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

# data pre-process
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)/255
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)/255
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)/255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)/255
    input_shape = (28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

inputs = Input(shape=input_shape)
conv_1 = Conv2D(6, kernel_size=5, strides=1, padding='same', activation='sigmoid')(inputs)
conv_1 = MaxPooling2D((2, 2), strides=2)(conv_1)
conv_2 = Conv2D(16, kernel_size=5, strides=1, padding='valid', activation='sigmoid')(conv_1)
conv_2 = MaxPooling2D((2, 2), strides=2)(conv_2)
dense_1 = Flatten()(conv_2)
dense_1 = Dense(120)(dense_1)
dense_2 = Dense(84)(dense_1)
dense_3 = Dense(10, activation='softmax')(dense_2)
model = Model(inputs=inputs, outputs=dense_3)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(X_train, Y_train, epochs=1, batch_size=32, validation_data=(X_test, Y_test))
