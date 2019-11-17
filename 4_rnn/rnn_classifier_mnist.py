
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, SimpleRNN
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

# data pre-process
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

# build model
model = Sequential()
model.add(SimpleRNN(input_dim=28, input_length=28, output_dim=50, unroll=True))
model.add(Dense(10, activation='softmax'))
adam = Adam(0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# train
model.fit(X_train, Y_train, epochs=2, batch_size=32, validation_data=(X_test, Y_test))
