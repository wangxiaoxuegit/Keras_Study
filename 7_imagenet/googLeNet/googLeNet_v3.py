
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


# inception_v2_1 figure-5
def inception_module_v2_1(inputs,
                          filters_3x3_reduce_1, filters_3x3_1_1, filters_3x3_1_2,
                          filters_3x3_reduce_2, filters_3x3_2,
                          filters_pool_proj,
                          filters_1x1):
    conv_3x3_reduce_1 = Conv2D(filters=filters_3x3_reduce_1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_3x3_1_1 = Conv2D(filters=filters_3x3_1_1, kernel_size=3, strides=1, padding='same', activation='relu')(conv_3x3_reduce_1)
    conv_3x3_1_2 = Conv2D(filters=filters_3x3_1_2, kernel_size=3, strides=1, padding='same', activation='relu')(conv_3x3_1_1)
    conv_3x3_reduce_2 = Conv2D(filters=filters_3x3_reduce_2, kernel_size=1, strides=1, padding='same',activation='relu')(inputs)
    conv_3x3_2 = Conv2D(filters=filters_3x3_2, kernel_size=1, strides=1, padding='same', activation='relu')(conv_3x3_reduce_2)
    averagepool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_pool_proj = Conv2D(filters=filters_pool_proj, kernel_size=1, strides=1, padding='same', activation='relu')(averagepool)
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    outputs = concatenate([conv_3x3_1_2, conv_3x3_2, conv_pool_proj, conv_1x1], axis=3)
    return outputs


# inception_v2_2 figure-6
def inception_module_v2_2(inputs,
                          filter_size,
                          filters_1_1x1, filters_1_1xn_1, filters_1_nx1_1, filters_1_1xn_2, filters_1_nx1_2,
                          filters_2_1x1, filters_2_1xn, filters_2_nx1,
                          filters_3_pool_1x1,
                          filters_4_1x1):
    conv_1_1x1 = Conv2D(filters=filters_1_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_1_1xn_1 = Conv2D(filters=filters_1_1xn_1, kernel_size=(1, filter_size), strides=1, padding='same', activation='relu')(conv_1_1x1)
    conv_1_nx1_1 = Conv2D(filters=filters_1_nx1_1, kernel_size=(filter_size, 1), strides=1, padding='same', activation='relu')(conv_1_1xn_1)
    conv_1_1xn_2 = Conv2D(filters=filters_1_1xn_2, kernel_size=(1, filter_size), strides=1, padding='same', activation='relu')(conv_1_nx1_1)
    conv_1_nx1_2 = Conv2D(filters=filters_1_nx1_2, kernel_size=(filter_size, 1), strides=1, padding='same', activation='relu')(conv_1_1xn_2)
    conv_2_1x1 = Conv2D(filters=filters_2_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_2_1xn = Conv2D(filters=filters_2_1xn, kernel_size=(1, filter_size), strides=1, padding='same',activation='relu')(conv_2_1x1)
    conv_2_nx1 = Conv2D(filters=filters_2_nx1, kernel_size=(filter_size, 1), strides=1, padding='same',activation='relu')(conv_2_1xn)
    averagepool_3_3x3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_3_1x1 = Conv2D(filters=filters_3_pool_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(averagepool_3_3x3)
    conv_4_1x1 = Conv2D(filters=filters_4_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    outputs = concatenate([conv_1_nx1_2, conv_2_nx1, conv_3_1x1, conv_4_1x1], axis=3)
    return outputs


# inception_v2_3 figure-7
def inception_module_v2_3(inputs,
                          filters_1_1x1, filters_1_3x3, filters_1_1x3, filters_1_3x1,
                          filters_2_1x1, filters_2_1x3, filters_2_3x1,
                          filters_3_pool_1x1,
                          filters_4_1x1):
    conv_1_1x1 = Conv2D(filters=filters_1_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_1_3x3 = Conv2D(filters=filters_1_3x3, kernel_size=1, strides=1, padding='same', activation='relu')(conv_1_1x1)
    conv_1_1x3 = Conv2D(filters=filters_1_1x3, kernel_size=1, strides=1, padding='same', activation='relu')(conv_1_3x3)
    conv_1_3x1 = Conv2D(filters=filters_1_3x1, kernel_size=1, strides=1, padding='same', activation='relu')(conv_1_3x3)
    conv_2_1x1 = Conv2D(filters=filters_2_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_2_1x3 = Conv2D(filters=filters_2_1x3, kernel_size=1, strides=1, padding='same', activation='relu')(conv_2_1x1)
    conv_2_3x1 = Conv2D(filters=filters_2_3x1, kernel_size=1, strides=1, padding='same', activation='relu')(conv_2_1x1)
    averagepool_3_3x3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_3_1x1 = Conv2D(filters=filters_3_pool_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(averagepool_3_3x3)
    conv_4_1x1 = Conv2D(filters=filters_4_1x1, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    outputs = concatenate([conv_1_1x3, conv_1_3x1, conv_2_1x3, conv_2_3x1, conv_3_1x1, conv_4_1x1], axis=3)
    return outputs


'''
wide = (width + 2 * padding - kernel_size) / stride + 1
height = (height + 2 * padding - kernel_size) / stride + 1
'''


def googLeNet_v3(weights_path=None):
    inputs = Input(shape=(299, 299, 3))
    conv_1_3x3 = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation='relu')(inputs)
    conv_2_3x3 = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_1_3x3)
    conv_3_3x3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(conv_2_3x3)
    maxpool_1_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv_3_3x3)
    conv_4_3x3 = Conv2D(filters=80, kernel_size=3, strides=1, padding='valid', activation='relu')(maxpool_1_3x3)
    conv_5_3x3 = Conv2D(filters=192, kernel_size=3, strides=2, padding='valid', activation='relu')(conv_4_3x3)
    conv_6_3x3 = Conv2D(filters=288, kernel_size=3, strides=1, padding='valid', activation='relu')(conv_5_3x3)

    inception_1_1 = inception_module_v2_1(inputs=conv_6_3x3,
                                          filters_3x3_reduce_1=64, filters_3x3_1_1=96, filters_3x3_1_2=96,
                                          filters_3x3_reduce_2=48, filters_3x3_2=64,
                                          filters_pool_proj=32,
                                          filters_1x1=64)
    inception_1_2 = inception_module_v2_1(inputs=inception_1_1,
                                          filters_3x3_reduce_1=64, filters_3x3_1_1=96, filters_3x3_1_2=96,
                                          filters_3x3_reduce_2=48, filters_3x3_2=64,
                                          filters_pool_proj=64,
                                          filters_1x1=64)
    inception_1_3 = inception_module_v2_1(inputs=inception_1_2,
                                          filters_3x3_reduce_1=64, filters_3x3_1_1=96, filters_3x3_1_2=96,
                                          filters_3x3_reduce_2=48, filters_3x3_2=64,
                                          filters_pool_proj=64,
                                          filters_1x1=64)

    inception_2_1 = inception_module_v2_2(inputs=inception_1_3, filter_size=7,
                                          filters_1_1x1=128, filters_1_1xn_1=128, filters_1_nx1_1=128,
                                          filters_1_1xn_2=128, filters_1_nx1_2=192,
                                          filters_2_1x1=128, filters_2_1xn=128, filters_2_nx1=192,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=192)
    inception_2_2 = inception_module_v2_2(inputs=inception_2_1, filter_size=7,
                                          filters_1_1x1=128, filters_1_1xn_1=128, filters_1_nx1_1=128,
                                          filters_1_1xn_2=128, filters_1_nx1_2=192,
                                          filters_2_1x1=128, filters_2_1xn=128, filters_2_nx1=192,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=192)
    inception_2_3 = inception_module_v2_2(inputs=inception_2_2, filter_size=7,
                                          filters_1_1x1=128, filters_1_1xn_1=128, filters_1_nx1_1=128,
                                          filters_1_1xn_2=128, filters_1_nx1_2=192,
                                          filters_2_1x1=128, filters_2_1xn=128, filters_2_nx1=192,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=192)
    inception_2_4 = inception_module_v2_2(inputs=inception_2_3, filter_size=7,
                                          filters_1_1x1=128, filters_1_1xn_1=128, filters_1_nx1_1=128,
                                          filters_1_1xn_2=128, filters_1_nx1_2=192,
                                          filters_2_1x1=128, filters_2_1xn=128, filters_2_nx1=192,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=192)
    inception_2_5 = inception_module_v2_2(inputs=inception_2_4, filter_size=7,
                                          filters_1_1x1=128, filters_1_1xn_1=128, filters_1_nx1_1=128,
                                          filters_1_1xn_2=128, filters_1_nx1_2=192,
                                          filters_2_1x1=128, filters_2_1xn=128, filters_2_nx1=192,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=192)

    inception_3_1 = inception_module_v2_3(inputs=inception_2_5,
                                          filters_1_1x1=484, filters_1_3x3=384, filters_1_1x3=384, filters_1_3x1=384,
                                          filters_2_1x1=384, filters_2_1x3=384, filters_2_3x1=384,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=320)
    inception_3_2 = inception_module_v2_3(inputs=inception_3_1,
                                          filters_1_1x1=484, filters_1_3x3=384, filters_1_1x3=384, filters_1_3x1=384,
                                          filters_2_1x1=384, filters_2_1x3=384, filters_2_3x1=384,
                                          filters_3_pool_1x1=192,
                                          filters_4_1x1=320)

    maxpool_2_8x8 = MaxPooling2D(pool_size=(8, 8), strides=1, padding='valid')(inception_3_2)
    drop_1 = Dropout(0.4)(maxpool_2_8x8)
    dense_1 = Dense(1000, activation='softmax')(drop_1)

    model = Model(inputs=inputs, outputs=dense_1)
    if weights_path:
        model.load_weights(weights_path)
    return model


googlenet_v3 = googLeNet_v3()
googlenet_v3.summary()
plot_model(googlenet_v3, to_file='googlenet_v3.png')
