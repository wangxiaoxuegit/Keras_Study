
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 1.5 * X + 3 + np.random.normal(0, 0.05, (200, ))
# plt.scatter(X, Y)
# plt.show()
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# build model
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

# train model
print('-------------training-------------')
for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ',cost)
print('----------training finish----------')

# evaluate model
print("-------------evaluate-------------")
cost = model.evaluate(X_test, Y_test)
print('evaluate cost: ',cost)
w, b = model.get_weights()
print('weight: ', w, '\nbias: ', b)
print('----------evaluate finish----------')

# plot predict result
Y_predict = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_predict)
plt.show()
