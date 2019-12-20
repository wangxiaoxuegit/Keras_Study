
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model


def identity_block(x, kernel_size, filters):
    f1, f2, f3 = filters
    x_shortcut = x
    x = Conv2D(filters=f1, kernel_size=1, padding='valid', strides=1)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f2, kernel_size=kernel_size, padding='same', strides=1)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f3, kernel_size=1, padding='valid', strides=1)(x)
    x = BatchNormalization(axis=3)(x)
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


def conv_block(x, kernel_size, filters, s):
    f1, f2, f3 = filters
    x_shortcut = x
    x = Conv2D(filters=f1, kernel_size=1, padding='valid', strides=s)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f2, kernel_size=kernel_size, padding='same', strides=1)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f3, kernel_size=1, padding='valid', strides=1)(x)
    x = BatchNormalization(axis=3)(x)
    x_shortcut = Conv2D(filters=f3, kernel_size=1, padding='valid', strides=s)(x_shortcut)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=(224, 224, 3), classes=1000):
    # input
    inputs = Input(shape=input_shape)

    # stage_1
    conv_1 = Conv2D(filters=64, kernel_size=7, padding='same', strides=2)(inputs)
    bn_1 = BatchNormalization(axis=3)(conv_1)
    activ_1 = Activation('relu')(bn_1)
    maxpool_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(activ_1)

    # stage_2
    stage_2 = conv_block(maxpool_1, kernel_size=3, filters=(64, 64, 256), s=1)
    stage_2 = identity_block(stage_2, kernel_size=3, filters=(64, 64, 256))
    stage_2 = identity_block(stage_2, kernel_size=3, filters=(64, 64, 256))

    # stage_3
    stage_3 = conv_block(stage_2, kernel_size=3, filters=(128, 128, 512), s=2)
    stage_3 = identity_block(stage_3, kernel_size=3, filters=(128, 128, 512))
    stage_3 = identity_block(stage_3, kernel_size=3, filters=(128, 128, 512))
    stage_3 = identity_block(stage_3, kernel_size=3, filters=(128, 128, 512))

    # stage_4
    stage_4 = conv_block(stage_3, kernel_size=3, filters=(256, 256, 1024), s=2)
    stage_4 = identity_block(stage_4, kernel_size=3, filters=(256, 256, 1024))
    stage_4 = identity_block(stage_4, kernel_size=3, filters=(256, 256, 1024))
    stage_4 = identity_block(stage_4, kernel_size=3, filters=(256, 256, 1024))
    stage_4 = identity_block(stage_4, kernel_size=3, filters=(256, 256, 1024))
    stage_4 = identity_block(stage_4, kernel_size=3, filters=(256, 256, 1024))

    # stage_5
    stage_5 = conv_block(stage_4, kernel_size=3, filters=(512, 512, 2048), s=2)
    stage_5 = identity_block(stage_5, kernel_size=3, filters=(512, 512, 2048))
    stage_5 = identity_block(stage_5, kernel_size=3, filters=(512, 512, 2048))

    # stage_6
    avepool_1 = AveragePooling2D(pool_size=(2, 2), padding='same')(stage_5)
    flat_1 = Flatten()(avepool_1)
    dense_1 = Dense(classes, activation='softmax')(flat_1)

    model = Model(inputs=inputs, outputs=dense_1)
    return model


resnet50 = ResNet50()
resnet50.summary()
plot_model(resnet50, 'ResNet_50.png')

