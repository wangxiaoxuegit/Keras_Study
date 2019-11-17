
import matplotlib.pyplot as plt
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input

# data pre-process
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
x_train = (x_train.astype('float32') / 255) * 2 - 1
x_test = (x_test.astype('float32') / 255) * 2 - 1

# build model
encoding_dim = 2
inputs = Input(shape=(784,))
# encoder layers
encoded = Dense(128, activation='relu')(inputs)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)
# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)
autoencoder = Model(inputs=inputs, outputs=decoded)
encoder = Model(inputs=inputs, outputs=encoder_output)
autoencoder.compile(optimizer='adam', loss='mse')

# train
autoencoder.fit(x_train, x_train, epochs=10, batch_size=32, shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()
