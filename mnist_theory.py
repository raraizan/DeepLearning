import matplotlib.pyplot as plt 
import numpy as np
from keras.datasets import mnist

(X, y), (X_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=(5, 5))

for i in range(10):
    index = np.where(y == i)
    index = [
        np.random.choice(index[0]) for _ in range(10)
    ]
    for j, el in enumerate(index):
        ax = fig.add_subplot(
            10,
            10,
            i * 10 + j + 1,
        )
        ax.imshow(X[el], cmap='gray')
        ax.axis('off')

plt.show()