
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import RMSprop

# data pre-process
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)
print(Y_test[0])

# build model
model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
# optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy',metrics=['accuracy'])

# train model
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# evaluate
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)
