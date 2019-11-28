
from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from cuntomlayers import crosschannelnormalization
from cuntomlayers import Softmax4D


def VGG_16(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(None, None, 3))
    else:
        inputs = Input(shape=(224, 224, 3))

    conv_1_1 = ZeroPadding2D((1, 1))(inputs)
    conv_1_1 = Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_1)
    conv_1_1 = ZeroPadding2D((1, 1))(conv_1_1)
    conv_1_2 = Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_1)
    conv_1_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_1_2)

    conv_2_1 = ZeroPadding2D((1, 1))(conv_1_2)
    conv_2_1 = Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_2_1)
    conv_2_2 = ZeroPadding2D((1, 1))(conv_2_1)
    conv_2_2 = Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_2_2)
    conv_2_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2_2)

    conv_3_1 = ZeroPadding2D((1, 1))(conv_2_2)
    conv_3_1 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_1)
    conv_3_2 = ZeroPadding2D((1, 1))(conv_3_1)
    conv_3_2 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_2)
    conv_3_3 = ZeroPadding2D((1, 1))(conv_3_2)
    conv_3_3 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_3)
    conv_3_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv_3_3)

    conv_4_1 = ZeroPadding2D((1, 1))(conv_3_3)
    conv_4_1 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_1)
    conv_4_2 = ZeroPadding2D((1, 1))(conv_4_1)
    conv_4_2 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_2)
    conv_4_3 = ZeroPadding2D((1, 1))(conv_4_2)
    conv_4_3 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_3)
    conv_4_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv_4_3)

    conv_5_1 = ZeroPadding2D((1, 1))(conv_4_3)
    conv_5_1 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_1)
    conv_5_2 = ZeroPadding2D((1, 1))(conv_5_1)
    conv_5_2 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_2)
    conv_5_3 = ZeroPadding2D((1, 1))(conv_5_2)
    conv_5_3 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_3)
    conv_5_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv_5_3)

    if heatmap:
        dense_1 = Conv2D(4096, kernel_size=7, strides=1, padding='valid', activation='relu')(conv_5_3)
        dense_2 = Conv2D(4096, kernel_size=1, strides=1, padding='valid', activation='relu')(dense_1)
        dense_3 = Conv2D(1000, kernel_size=1, strides=1, padding='valid')(dense_2)
        outputs = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten()(conv_5_3)
        dense_1 = Dense(4096, activation='relu')(dense_1)
        dense_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu')(dense_1)
        dense_2 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000)(dense_2)
        outputs = Activation('softmax')(dense_3)

    model = Model(inputs=inputs, outputs=outputs)
    if weights_path:
        model.load_weights(weights_path)
    return model


def VGG_19(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(None, None, 3))
    else:
        inputs = Input(shape=(224, 224, 3))

    conv_1_1 = ZeroPadding2D((1, 1))(inputs)
    conv_1_1 = Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_1)
    conv_1_1 = ZeroPadding2D((1, 1))(conv_1_1)
    conv_1_2 = Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_1)
    conv_1_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_1_2)

    conv_2_1 = ZeroPadding2D((1, 1))(conv_1_2)
    conv_2_1 = Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_2_1)
    conv_2_2 = ZeroPadding2D((1, 1))(conv_2_1)
    conv_2_2 = Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_2_2)
    conv_2_2 = MaxPooling2D((2, 2), strides=(2, 2))(conv_2_2)

    conv_3_1 = ZeroPadding2D((1, 1))(conv_2_2)
    conv_3_1 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_1)
    conv_3_2 = ZeroPadding2D((1, 1))(conv_3_1)
    conv_3_2 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_2)
    conv_3_3 = ZeroPadding2D((1, 1))(conv_3_2)
    conv_3_3 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_3)
    conv_3_4 = ZeroPadding2D((1, 1))(conv_3_3)
    conv_3_4 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3_4)
    conv_3_4 = MaxPooling2D((2, 2), strides=(2, 2))(conv_3_4)

    conv_4_1 = ZeroPadding2D((1, 1))(conv_3_4)
    conv_4_1 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_1)
    conv_4_2 = ZeroPadding2D((1, 1))(conv_4_1)
    conv_4_2 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_2)
    conv_4_3 = ZeroPadding2D((1, 1))(conv_4_2)
    conv_4_3 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_3)
    conv_4_4 = ZeroPadding2D((1, 1))(conv_4_3)
    conv_4_4 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_4)
    conv_4_4 = MaxPooling2D((2, 2), strides=(2, 2))(conv_4_4)

    conv_5_1 = ZeroPadding2D((1, 1))(conv_4_4)
    conv_5_1 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_1)
    conv_5_2 = ZeroPadding2D((1, 1))(conv_5_1)
    conv_5_2 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_2)
    conv_5_3 = ZeroPadding2D((1, 1))(conv_5_2)
    conv_5_3 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_3)
    conv_5_4 = ZeroPadding2D((1, 1))(conv_5_3)
    conv_5_4 = Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_4)
    conv_5_4 = MaxPooling2D((2, 2), strides=(2, 2))(conv_5_4)

    if heatmap:
        dense_1 = Conv2D(4096, kernel_size=7, strides=1, padding='valid', activation='relu')(conv_5_4)
        dense_2 = Conv2D(4096, kernel_size=1, strides=1, padding='valid', activation='relu')(dense_1)
        dense_3 = Conv2D(1000, kernel_size=1, strides=1, padding='valid')(dense_2)
        outputs = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten()(conv_5_4)
        dense_1 = Dense(4096, activation='relu')(dense_1)
        dense_1 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu')(dense_1)
        dense_2 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000)(dense_2)
        outputs = Activation('softmax')(dense_3)

    model = Model(inputs=inputs, outputs=outputs)
    if weights_path:
        model.load_weights(weights_path)
    return model


vgg16 = VGG_16()
vgg16.summary()

vgg19 = VGG_19()
vgg19.summary()
