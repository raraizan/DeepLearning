from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation 
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 

import numpy as np   
(X, y), (Xtest, ytest) = mnist.load_data()

# Visualizar algunas imagenes de la base de datos
plt.figure(figsize = (7,7))
for i in range(10):
    index = np.where(y == i)
    index = index[0][np.random.randint(index[0].shape[0], size=10)]
    for j in range(10):
        plt.subplot(10,10,i*10+j+1)
        plt.imshow(X[index[j]], cmap='gray', interpolation='none')
        plt.axis('off')
plt.show()