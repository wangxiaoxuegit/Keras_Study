
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

'''
wide = (224 + 2 * padding - kernel_size) / stride + 1
height = (224 + 2 * padding - kernel_size) / stride + 1
'''


def AlexNet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(None, None, 3))
    else:
        inputs = Input(shape=(227, 227, 3))

    conv_1 = Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu')(inputs)
    # conv_1 = crosschannelnormalization()(conv_1)
    conv_1 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)

    conv_2 = ZeroPadding2D((2, 2))(conv_1)
    conv_2 = Conv2D(256, kernel_size=5, strides=1, padding='valid', activation='relu')(conv_2)
    # conv_2 = crosschannelnormalization()(conv_2)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)

    conv_3 = ZeroPadding2D((1, 1))(conv_2)
    conv_3 = Conv2D(384, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Conv2D(384, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4)

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5)

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2))(conv_5)
    if heatmap:
        dense_1 = Conv2D(4096, kernel_size=6, strides=1, padding='valid', activation='relu')(dense_1)
        dense_2 = Conv2D(4096, kernel_size=1, strides=1, padding='valid', activation='relu')(dense_1)
        dense_3 = Conv2D(1000, kernel_size=1, strides=1, padding='valid')(dense_2)
        dense_3 = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten()(dense_1)
        dense_1 = Dense(4096, activation='relu')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, activation='softmax')(dense_3)

    model = Model(inputs=inputs, outputs=dense_3)
    if weights_path:
        model.load_weights(weights_path)
    return model


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

