#@title Default title text
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, ZeroPadding1D, AveragePooling1D,MaxPooling1D
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.callbacks import History 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import DataProcessor as dp
from contextlib import redirect_stdout


#%% plot the training loss and accuracy
def plotResults(epochs,history): 
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")

    # make the graph understandable: 
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    return plt

#%% Data modifying
data_df = pd.read_csv('./newdata/2019.csv')
X = data_df.drop(['Hit'],axis=1)
Y = data_df['Hit']
x_train, x_val, y_train, y_val = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.3, shuffle= True)

x_scaler = MinMaxScaler() 
x_scaler.fit(x_train)
x_train_norm = x_scaler.transform(x_train)
x_val_norm = x_scaler.transform(x_val)

num_classes = 2
input_shape = (13,)
#print(input_shape)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_val_binary = keras.utils.to_categorical(y_val, num_classes)
x_train_norm = x_train_norm.reshape(x_train_norm.shape[0], 13,1)
x_val_norm = x_val_norm.reshape(x_val_norm.shape[0], 13,1)

#%% define model
CNN = Sequential()
# layer 1
CNN.add(Conv1D(128, (2), input_shape=(13,1),activation='relu'))
CNN.add(MaxPooling1D())
CNN.add(Dropout(0.5))
# layer 2
CNN.add(Conv1D(256,(2),activation='relu'))
CNN.add(MaxPooling1D(2))
CNN.add(Dropout(0.5))
# layer 3
# CNN.add(Conv1D(128, (1),activation='relu'))
# CNN.add(MaxPooling1D(2))
# CNN.add(Dropout(0.5))
# CNN.add(Conv1D(64,(2),activation='relu'))
# layer 4
CNN.add(Flatten())
# Dense layer
CNN.add(Dense(64,activation='relu'))
CNN.add(Dense(16, activation='relu'))
CNN.add(Dense(num_classes, activation='softmax'))

CNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(CNN.summary())

#%% model training
batch_size = 20
epochs = 20
history = History()
CNN.fit(x_train_norm, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[history],
          validation_data=(x_val_norm, y_val_binary))


#%% testing
# test_df = pd.read_csv('./2020.csv')
# X_test = test_df.drop(['Hit'],axis=1)
# Y_test = test_df['Hit']
# x_test_norm = x_scaler.transform(X_test)
# x_test_norm = x_test_norm.reshape(x_test_norm.shape[0], 13,1)
# y_test_binary = keras.utils.to_categorical(Y_test, num_classes)
# score = CNN.evaluate(x_test_norm, y_test_binary, batch_size=20, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

#%%save model summary
from contextlib import redirect_stdout
with open('./result/CNN/modelsummary3.txt', 'w') as f:
    with redirect_stdout(f):
        CNN.summary()

#save training history
pd.DataFrame.from_dict(history.history).to_csv('./result/CNN/CNNhistory3.csv')

#%% plot the training loss and accuracy and save it
TrainLossAcc=plotResults(epochs,history)
TrainLossAcc.savefig('./result/CNN/CNNTrainLossAcc3.png')

#%% predict labels and evalutate
y_pred=CNN.predict(x_val_norm)
y_pred = np.argmax(y_pred, axis=-1)
report,rocfig=dp.evaluate_on_training_set(y_val,y_pred)
pred_fig=dp.plot_pred_original(y_pred,y_val,'CNN')
print(report)

#%% save predict result
rocfig.savefig('./result/CNN/CNNroc3.png')
with open('./result/CNN/CNNreport3', 'w') as f:
    [f.write('{0}:\n{1}\n'.format(key, value)) for key, value in report.items()]

pred_fig.savefig('./result/CNN/CNNpred3.png')



#%% save the model
CNN.save('./trainedModel/CNN3')
# load model
# from tensorflow import keras
# model = keras.models.load_model('path/to/location')