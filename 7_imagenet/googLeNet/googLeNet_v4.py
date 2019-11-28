
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model


# stem
def stem(inputs):
    conv_1_3x3 = Conv2D(32, kernel_size=3, strides=2, padding='valid', activation='relu')(inputs)
    conv_2_3x3 = Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_3x3)
    conv_3_3x3 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(conv_2_3x3)
    conv_4_3x3 = Conv2D(96, kernel_size=3, strides=2, padding='valid', activation='relu')(conv_3_3x3)
    maxpool_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv_3_3x3)
    concat_1 = concatenate([conv_4_3x3, maxpool_1], axis=3)
    conv_branch1_1x1 = Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(concat_1)
    conv_branch1_3x3 = Conv2D(96, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_branch1_1x1)
    conv_branch2_1x1 = Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(concat_1)
    conv_branch2_7x1 = Conv2D(96, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x1)
    conv_branch2_1x7 = Conv2D(64, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv_branch2_7x1)
    conv_branch2_3x3 = Conv2D(96, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_branch2_1x7)
    concat_2 = concatenate([conv_branch1_3x3, conv_branch2_3x3], axis=3)
    conv_5_3x3 = Conv2D(192, kernel_size=3, strides=2, padding='valid', activation='relu')(concat_2)
    maxpool_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(concat_2)
    concat_3 = concatenate([conv_5_3x3, maxpool_2], axis=3)
    return concat_3


# inception-a
def inception_v4_a(inputs):
    conv_branch1_1x1 =  Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch1_3x3_1 = Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu')(conv_branch1_1x1)
    conv_branch1_3x3_2 = Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu')(conv_branch1_3x3_1)
    conv_branch2_1x1 = Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch2_3x3 = Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu')(conv_branch2_1x1)
    conv_branch3_1x1 = Conv2D(96, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    pool_branch4_3x3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_branch4_1x1 = Conv2D(96, kernel_size=1, strides=1, padding='same', activation='relu')(pool_branch4_3x3)
    outputs = concatenate([conv_branch1_3x3_2, conv_branch2_3x3, conv_branch3_1x1, conv_branch4_1x1], axis=3)
    return outputs


# inception-b
def inception_v4_b(inputs):
    conv_branch1_1x1 =  Conv2D(192, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch2_1x7_1 = Conv2D(224, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv_branch1_1x1)
    conv_branch2_7x1_1 = Conv2D(224, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x7_1)
    conv_branch2_1x7_2 = Conv2D(256, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv_branch2_7x1_1)
    conv_branch2_7x1_2 = Conv2D(256, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x7_2)
    conv_branch2_1x1 = Conv2D(192, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch2_1x7 = Conv2D(224, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv_branch2_1x1)
    conv_branch2_7x1 = Conv2D(256, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x7)
    conv_branch3_1x1 = Conv2D(384, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    pool_branch4_3x3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_branch4_1x1 = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(pool_branch4_3x3)
    outputs = concatenate([conv_branch2_7x1_2, conv_branch2_7x1, conv_branch3_1x1, conv_branch4_1x1], axis=3)
    return outputs


# inception-c
def inception_v4_c(inputs):
    conv_branch1_1x1 =  Conv2D(384, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch2_1x3_1 = Conv2D(448, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv_branch1_1x1)
    conv_branch2_3x1_1 = Conv2D(512, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x3_1)
    conv_branch2_1x3_2 = Conv2D(256, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv_branch2_3x1_1)
    conv_branch2_3x1_2 = Conv2D(256, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv_branch2_3x1_1)
    conv_branch2_1x1 = Conv2D(384, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_branch2_1x3 = Conv2D(356, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv_branch2_1x1)
    conv_branch2_3x1 = Conv2D(256, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv_branch2_1x1)
    conv_branch3_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    pool_branch4_3x3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_branch4_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(pool_branch4_3x3)
    outputs = concatenate([conv_branch2_1x3_2, conv_branch2_3x1_2, conv_branch2_1x3, conv_branch2_3x1, conv_branch3_1x1, conv_branch4_1x1], axis=3)
    return outputs


# reduction-a
def reduction_a(inputs, filters_branch2_3x3,
                filters_branch3_1x1, filters_branch3_3x3_1, filters_branch3_3x3_2):
    pool_branch1_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(inputs)
    conv_branch2_3x3 = Conv2D(filters_branch2_3x3, kernel_size=3, strides=2, padding='valid')(inputs)
    conv_branch3_1x1 = Conv2D(filters_branch3_1x1, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch3_3x3_1 = Conv2D(filters_branch3_3x3_1, kernel_size=3, strides=1, padding='same')(conv_branch3_1x1)
    conv_branch3_3x3_2 = Conv2D(filters_branch3_3x3_2, kernel_size=3, strides=2, padding='valid')(conv_branch3_3x3_1)
    outputs = concatenate([pool_branch1_3x3, conv_branch2_3x3, conv_branch3_3x3_2], axis=3)
    return outputs


# reduction-b
def reduction_b(inputs):
    pool_branch1_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(inputs)
    conv_branch2_1x1 = Conv2D(192, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch2_3x3 = Conv2D(192, kernel_size=3, strides=2, padding='valid')(conv_branch2_1x1)
    conv_branch3_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch3_1x7 = Conv2D(256, kernel_size=(1, 7), strides=1, padding='same')(conv_branch3_1x1)
    conv_branch3_7x1 = Conv2D(320, kernel_size=(7, 1), strides=1, padding='same')(conv_branch3_1x7)
    conv_branch3_3x3 = Conv2D(320, kernel_size=3, strides=2, padding='valid')(conv_branch3_7x1)
    outputs = concatenate([pool_branch1_3x3, conv_branch2_3x3, conv_branch3_3x3], axis=3)
    return outputs


# googLeNet-v4
def googLeNet_v4(weights_path=None):
    inputs = Input(shape=(299, 299, 3))

    stem_1 = stem(inputs)

    inception_a_1 = inception_v4_a(stem_1)
    inception_a_2 = inception_v4_a(inception_a_1)
    inception_a_3 = inception_v4_a(inception_a_2)
    inception_a_4 = inception_v4_a(inception_a_3)

    reduction_a_1 = reduction_a(inputs=inception_a_4,filters_branch2_3x3=384,
                                filters_branch3_1x1=192, filters_branch3_3x3_1=224, filters_branch3_3x3_2=256)


    inception_b_1 = inception_v4_b(reduction_a_1)
    inception_b_2 = inception_v4_b(inception_b_1)
    inception_b_3 = inception_v4_b(inception_b_2)
    inception_b_4 = inception_v4_b(inception_b_3)
    inception_b_5 = inception_v4_b(inception_b_4)
    inception_b_6 = inception_v4_b(inception_b_5)
    inception_b_7 = inception_v4_b(inception_b_6)

    reduction_b_1 = reduction_b(inception_b_7)

    inception_c_1 = inception_v4_c(reduction_b_1)
    inception_c_2 = inception_v4_c(inception_c_1)
    inception_c_3 = inception_v4_c(inception_c_2)

    avg_pool_1 = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(inception_c_3)
    drop_1 = Dropout(0.8)(avg_pool_1)
    dense_1 = Dense(1000, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=dense_1)
    if weights_path:
        model.load_weights(weights_path)
    return model


googlenet_v4 = googLeNet_v4()
googlenet_v4.summary()
plot_model(googlenet_v4, to_file='googlenet_v4.png')

