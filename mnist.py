import matplotlib.pyplot as plt 
import numpy as np
from keras.datasets import mnist

(X, y), (X_test, y_test) = mnist.load_data()

X_train = X[:50000,:,:]
y_train = y[:50000]

X_val = X[50000:,:,:]
y_val = y[50000:]

hist_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
hist_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
hist_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for num in y_val:
    hist_val[num] += 1

for num in y_train:
    hist_train[num] += 1

for num in y_test:
    hist_test[num] += 1

print(hist_train)
print(hist_val)
print(hist_test)

n_bins = range(10)

fig = plt.figure(figsize = (12,4))
ax0 = fig.add_subplot(131)
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)

ax0.bar(n_bins, hist_train)
ax0.set_title("train")

ax1.bar(n_bins, hist_val)
ax1.set_title("validation")

ax2.bar(n_bins, hist_test)
ax2.set_title("test")

plt.show()