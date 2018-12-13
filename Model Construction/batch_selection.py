import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import timeit

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from numpy.random import seed
seed(99)
from cntk.cntk_py import set_fixed_random_seed
set_fixed_random_seed(99)

#start = timeit.default_timer()

times = pd.DataFrame()

def read_data(data):
    x = data.iloc[:, :26].values
    y = data.iloc[:, 26].values

    t = 571
    x = np.reshape(x, (int(x.shape[0]/t), t, x.shape[1]))
    
    indices = [ind for ind in range(len(y)) if ind%t==0]
    y = y[indices]
    
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_new = label_encoder.fit_transform(y)
    y = y_new

    from sklearn.preprocessing import LabelBinarizer
    multi_label = LabelBinarizer()
    y_new = multi_label.fit_transform(y)
    y = y_new

    return x, y

def model_fit(x, y, n_neurons, n_epochs, n_batch_size):
    start = timeit.default_timer()
    
    model = Sequential()
    model.add(SimpleRNN(n_neurons, activation='relu', input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Fit the model
    history = model.fit(x, y, epochs=n_epochs, batch_size=n_batch_size)
    loss = history.history['loss']
    acc = history.history['acc']
    stop = timeit.default_timer()
    times = stop - start
    return loss, acc, times

def experiment():
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    date = "20181210"
    n_batch = [12, 24, 36, 48]
    neuron = 12
    n_epochs = 100
    x, y = read_data(dataset)
    loss_results = pd.DataFrame()
    acc_results = pd.DataFrame()
    times = pd.DataFrame()
    for batch in n_batch:
        loss_results[batch], acc_results[batch], timer = model_fit(x, y, neuron, n_epochs, batch)
        times.loc[0, batch] = timer
    loss_results.to_csv(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_DATA_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    #plot loss
    plt.figure()
    for loss in n_batch:
        plt.plot(loss_results[loss])

    plt.title('batch size %s' %(n_batch))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(n_batch, loc='upper right')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    print(times)

def experiment_1():
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    date = "20181210"
    n_batch = [60]
    neuron = 12
    n_epochs = 100
    x, y = read_data(dataset)
    loss_results = pd.DataFrame()
    acc_results = pd.DataFrame()
    times = pd.DataFrame()
    for batch in n_batch:
        loss_results[batch], acc_results[batch], timer = model_fit(x, y, neuron, n_epochs, batch)
        times.loc[0, batch] = timer
    loss_results.to_csv(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_DATA_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    #plot loss
    plt.figure()
    for loss in n_batch:
        plt.plot(loss_results[loss])

    plt.title('batch size %s' %(n_batch))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(n_batch, loc='upper right')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    print(times)

def experiment_2():
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    date = "20181210"
    n_batch = [108, 120, 132, 144]
    neuron = 12
    n_epochs = 100
    x, y = read_data(dataset)
    loss_results = pd.DataFrame()
    acc_results = pd.DataFrame()
    times = pd.DataFrame()
    for batch in n_batch:
        loss_results[batch], acc_results[batch], timer = model_fit(x, y, neuron, n_epochs, batch)
        times.loc[0, batch] = timer
    loss_results.to_csv(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_DATA_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    #plot loss
    plt.figure()
    for loss in n_batch:
        plt.plot(loss_results[loss])

    plt.title('batch size %s' %(n_batch))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(n_batch, loc='upper right')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_BATCH_SIZE_TUNING_RNN_LOSS_E%s_B%s_N%s.png' %(date, n_epochs, n_batch, neuron))
    print(times)


#experiment()
experiment_1()
#experiment_2()


