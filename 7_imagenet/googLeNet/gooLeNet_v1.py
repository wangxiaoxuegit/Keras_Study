
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

'''
wide = (width + 2 * padding - kernel_size) / stride + 1
height = (height + 2 * padding - kernel_size) / stride + 1
'''

# inception_v1
def inception_module_v1(inputs,
                        filters_1x1,
                        filters_3x3_reduce, filters_3x3,
                        filters_5x5_reduce, filters_5x5,
                        filters_pool_proj):
    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=1, strides=1, padding='same',activation='relu')(inputs)
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=1, strides=1, padding='same',activation='relu')(inputs)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=1, strides=1, padding='same', activation='relu')(conv_3x3_reduce)
    conv_5x5_reduce= Conv2D(filters=filters_5x5_reduce, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=1, strides=1, padding='same', activation='relu')(conv_5x5_reduce)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(inputs)
    conv_pool_proj = Conv2D(filters=filters_pool_proj, kernel_size=1, strides=1, padding='same', activation='relu')(maxpool)
    outputs = concatenate([conv_1x1, conv_3x3, conv_5x5, conv_pool_proj], axis=3)
    return outputs


def googLeNet_v1(weights_path=None):
    inputs = Input(shape=(224, 224, 3))
    conv_1_7x7 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    maxpool_1_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv_1_7x7)

    conv_2_3x3_reduce = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(maxpool_1_3x3)
    conv_2_3x3 = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu')(conv_2_3x3_reduce)
    maxpool_2_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv_2_3x3)

    inception_3a = inception_module_v1(inputs=maxpool_2_3x3, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                                       filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
    inception_3b = inception_module_v1(inputs=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                                       filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
    maxpool_3_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(inception_3b)

    inception_4a = inception_module_v1(inputs=maxpool_3_3x3, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                                       filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)
    inception_4b = inception_module_v1(inputs=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                                       filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4c = inception_module_v1(inputs=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                                       filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4d = inception_module_v1(inputs=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                                       filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)
    inception_4e = inception_module_v1(inputs=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                       filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    maxpool_4_3x3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(inception_4e)

    inception_5a = inception_module_v1(inputs=maxpool_4_3x3, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                       filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    inception_5b = inception_module_v1(inputs=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
                                       filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)
    averagepool_1_7x7 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(inception_5b)

    drop_1 = Dropout(0.4)(averagepool_1_7x7)
    flaten_1 = Flatten()(drop_1)
    dense_1 = Dense(1000, activation='softmax')(flaten_1)
    model = Model(inputs=inputs, outputs=dense_1)

    if weights_path:
        model.load_weights(weights_path)
    return model


googlenet_v1 = googLeNet_v1()
googlenet_v1.summary()
plot_model(googlenet_v1, to_file='googlenet_v1.png')
