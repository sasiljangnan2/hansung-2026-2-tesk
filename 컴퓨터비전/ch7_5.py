# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:45:22 2023

@author: BigData
"""
#%%
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#%%
import tensorflow as tf
import tensorflow.keras.datasets as ds
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
#%% Data handling
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000, 784)
x_test=x_test.reshape(10000, 784)
x_train=x_train.astype(np.float32)/255.0  #0~1 normalization
x_test=x_test.astype(np.float32)/255.0 
y_train=tf.keras.utils.to_categorical(y_train, 10)
y_test=tf.keras.utils.to_categorical(y_test, 10)

#%% ANN model Construction
dmlp=Sequential()
dmlp.add(Input(shape=(784,)))
dmlp.add(Dense(units=1024, activation="relu"))
dmlp.add(Dense(units=512, activation="relu"))
dmlp.add(Dense(units=512, activation="relu"))
dmlp.add(Dense(units=10, activation="softmax"))

#%% Adam
dmlp.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), 
             metrics=["accuracy"]) #Learing environment configuration
hist=dmlp.fit(x_train, y_train, batch_size=128, epochs=50, 
         validation_data=(x_test, y_test), verbose=2)

print("Acuuracy", dmlp.evaluate(x_test, y_test, verbose=0)[1]*100)

dmlp.save("dmlp_trained.keras")

#%% Accuracy
import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"]) #default color: cyan
plt.plot(hist.history["val_accuracy"])
plt.title("Accuracy Graph")
plt.xlabel("epocs")
plt.ylabel("accuracy")
plt.legend(["train", "test"])
plt.grid()
plt.show()

#%% Loss

plt.plot(hist.history["loss"]) #default color: cyan
plt.plot(hist.history["val_loss"])
plt.title("Loss Graph")
plt.xlabel("epocs")
plt.ylabel("Loss")
plt.legend(["train", "test"])
plt.grid()
plt.show()
#%%
#CIFAR-10 에 대해서 실습해 볼 것 : 프로그램 7-6