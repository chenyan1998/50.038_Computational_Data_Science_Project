import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, ZeroPadding1D, AveragePooling1D
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.callbacks import History 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%% Data modifying
data_df1 = pd.read_csv('./01.csv')
data_df2 = pd.read_csv('./02.csv')
data_df3 = pd.read_csv('./03.csv')
data_df=pd.concat([data_df1,data_df2,data_df3],ignore_index=True)
X = data_df.drop(['Hit'],axis=1)
Y = data_df['Hit']
x_train, x_val, y_train, y_val = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.3, shuffle= True)

x_scaler = MinMaxScaler() 
x_scaler.fit(x_train)
x_train_norm = x_scaler.transform(x_train)
x_val_norm = x_scaler.transform(x_val)

print(x_train_norm.shape)
print(x_train_norm)
num_classes = 2
input_shape = (13,)
#print(input_shape)
# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_val_binary = keras.utils.to_categorical(y_val, num_classes)
x_train_norm = x_train_norm.reshape(x_train_norm.shape[0], 13,1)
#print(x_train)
x_val_norm = x_val_norm.reshape(x_val_norm.shape[0], 13,1)

#%%
# create and fit the LSTM network
from keras.layers import LSTM
RNN = Sequential()
RNN.add(LSTM(128, input_shape=(13,1)))
RNN.add(Dense(2))
RNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(RNN.summary())

Rhistory = History()
batch_size = 20
epochs = 30
RNN.fit(x_train_norm, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[Rhistory],
          validation_data=(x_val_norm, y_val_binary))

#%%
test_df = pd.read_csv('./03.csv')
X_test = test_df.drop(['Hit'],axis=1)
Y_test = test_df['Hit']
x_test_norm = x_scaler.transform(X_test)
x_test_norm = x_test_norm.reshape(x_test_norm.shape[0], 13,1)
y_test_binary = keras.utils.to_categorical(Y_test, num_classes)
score = CNN.evaluate(x_test_norm, y_test_binary, batch_size=20, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#%%
RNN.save('./RNN')