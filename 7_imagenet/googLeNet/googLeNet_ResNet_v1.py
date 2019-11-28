from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Add
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model


# stem
def stem(inputs):
    conv_1_3x3 = Conv2D(32, kernel_size=3, strides=2, padding='valid', activation='relu')(inputs)
    conv_2_3x3 = Conv2D(32, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_3x3)
    conv_3_3x3 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(conv_2_3x3)
    maxpool_1_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv_3_3x3)
    conv_4_1x1 = Conv2D(80, kernel_size=1, strides=1, padding='same', activation='relu')(maxpool_1_3x3)
    conv_5_3x3 = Conv2D(192, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_4_1x1)
    conv_6_3x3 = Conv2D(256, kernel_size=3, strides=2, padding='valid', activation='relu')(conv_5_3x3)
    return conv_6_3x3


# inception_resnet_a
def inception_resnet_a(inputs):
    actv_1 = Activation('relu')(inputs)
    conv_1_1x1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_1x1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_3_1x1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_3x3 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(conv_2_1x1)
    conv_3_3x3 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(conv_3_1x1)
    conv_4_3x3 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(conv_3_3x3)
    concat_1 = concatenate([conv_1_1x1, conv_2_3x3, conv_4_3x3], axis=3)
    conv_4_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='linear')(concat_1)
    add_1 = Add()([actv_1, conv_4_1x1])
    actv_2 = Activation('relu')(add_1)
    return actv_2


# inception_resnet_b
def inception_resnet_b(inputs):
    actv_1 = Activation('relu')(inputs)
    conv_1_1_1x1 = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_1_1x1 = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_2_1x7 = Conv2D(192, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(conv_2_1_1x1)
    conv_2_3_7x1 = Conv2D(192, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(conv_2_2_1x7)
    concat_1 = concatenate([conv_1_1_1x1, conv_2_3_7x1], axis=3)
    conv_1_2_1x1 = Conv2D(896, kernel_size=1, strides=1, padding='same', activation='linear')(concat_1)
    add_1 = Add()([actv_1, conv_1_2_1x1])
    actv_2 = Activation('relu')(add_1)
    return actv_2


# inception_resnet_c
def inception_resnet_c(inputs):
    actv_1 = Activation('relu')(inputs)
    conv_1_1_1x1 = Conv2D(192, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_1_1x1 = Conv2D(192, kernel_size=1, strides=1, padding='same', activation='relu')(actv_1)
    conv_2_2_1x3 = Conv2D(192, kernel_size=(1, 3), strides=1, padding='same', activation='relu')(conv_2_1_1x1)
    conv_2_3_3x1 = Conv2D(192, kernel_size=(3, 1), strides=1, padding='same', activation='relu')(conv_2_2_1x3)
    concat_1 = concatenate([conv_1_1_1x1, conv_2_3_3x1], axis=3)
    conv_1_2_1x1 = Conv2D(1792, kernel_size=1, strides=1, padding='same', activation='linear')(concat_1)
    add_1 = Add()([actv_1, conv_1_2_1x1])
    actv_2 = Activation('relu')(add_1)
    return actv_2


# reduction_a
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
    conv_branch2_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch2_3x3 = Conv2D(384, kernel_size=3, strides=2, padding='valid')(conv_branch2_1x1)
    conv_branch3_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch3_3x3 = Conv2D(256, kernel_size=3, strides=2, padding='valid')(conv_branch3_1x1)
    conv_branch4_1x1 = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputs)
    conv_branch4_3x3 = Conv2D(256, kernel_size=3, strides=1, padding='same')(conv_branch4_1x1)
    conv_branch4_3x3 = Conv2D(256, kernel_size=3, strides=2, padding='valid')(conv_branch4_3x3)
    outputs = concatenate([pool_branch1_3x3, conv_branch2_3x3, conv_branch3_3x3, conv_branch4_3x3], axis=3)
    return outputs


# googLeNet_ResNet_v1
def googLeNet_ResNet_v1(weights_path=None):
    inputs = Input(shape=(299, 299, 3))

    stem_1 = stem(inputs)

    inception_a_1 = inception_resnet_a(stem_1)
    inception_a_2 = inception_resnet_a(inception_a_1)
    inception_a_3 = inception_resnet_a(inception_a_2)
    inception_a_4 = inception_resnet_a(inception_a_3)
    inception_a_5 = inception_resnet_a(inception_a_4)

    reduction_a_1 = reduction_a(inputs=inception_a_4, filters_branch2_3x3=384,
                                filters_branch3_1x1=192, filters_branch3_3x3_1=192, filters_branch3_3x3_2=256)

    inception_b_1 = inception_resnet_b(reduction_a_1)
    inception_b_2 = inception_resnet_b(inception_b_1)
    inception_b_3 = inception_resnet_b(inception_b_2)
    inception_b_4 = inception_resnet_b(inception_b_3)
    inception_b_5 = inception_resnet_b(inception_b_4)
    inception_b_6 = inception_resnet_b(inception_b_5)
    inception_b_7 = inception_resnet_b(inception_b_6)
    inception_b_8 = inception_resnet_b(inception_b_7)
    inception_b_9 = inception_resnet_b(inception_b_8)
    inception_b_10 = inception_resnet_b(inception_b_9)

    reduction_b_1 = reduction_b(inception_b_10)

    inception_c_1 = inception_resnet_c(reduction_b_1)
    inception_c_2 = inception_resnet_c(inception_c_1)
    inception_c_3 = inception_resnet_c(inception_c_2)
    inception_c_4 = inception_resnet_c(inception_c_3)
    inception_c_5 = inception_resnet_c(inception_c_4)

    avg_pool_1 = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(inception_c_3)
    drop_1 = Dropout(0.8)(avg_pool_1)
    dense_1 = Dense(1000, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=dense_1)
    if weights_path:
        model.load_weights(weights_path)
    return model


googlenet_resnet_v1 = googLeNet_ResNet_v1()
googlenet_resnet_v1.summary()
plot_model(googlenet_resnet_v1, to_file='googlenet_resnet_v1.png')
