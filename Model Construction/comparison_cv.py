import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import timeit

from keras.models import Sequential
from keras import metrics
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from matplotlib import pyplot
from numpy.random import seed
seed(29)
from cntk.cntk_py import set_fixed_random_seed
set_fixed_random_seed(98)

start = timeit.default_timer()

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

def model_fit(x, y, n_neurons, n_epochs, n_batch_size, n_k_fold):

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_k_fold, shuffle=False, random_state=None)
    cvscores = []

    for train, test in kfold.split(x,y):
        model = Sequential()
        model.add(SimpleRNN(n_neurons, activation='relu', input_shape=(x.shape[1], x.shape[2])))
        model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
    
        #Fit the model
        model.fit(x[train], y[train], epochs=n_epochs, batch_size=n_batch_size)

        # Evaluate the model
        scores = model.evaluate(x[test], y[test], verbose=0)
        print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    return cvscores

def LSTM_model_fit(x, y, n_neurons, n_epochs, n_batch_size, n_k_fold):
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_k_fold, shuffle=False, random_state=None)
    cvscores = []

    for train, test in kfold.split(x,y):
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

        model.fit(x[train], y[train], epochs=n_epochs, batch_size=n_batch_size)
        scores = model.evaluate(x[test], y[test], verbose=0)
        print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    return cvscores

def GRU_model_fit(x, y, n_neurons, n_epochs, n_batch_size, n_k_fold):
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_k_fold, shuffle=False, random_state=None)
    cvscores = []

    for train, test in kfold.split(x,y):
        model = Sequential()
        model.add(GRU(n_neurons, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

        model.fit(x[train], y[train], epochs=n_epochs, batch_size=n_batch_size)
        scores = model.evaluate(x[test], y[test], verbose=0)
        print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    return cvscores

def experiment():
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    date = "20181203"
    neuron = 23
    n_epochs = 100
    n_batch = 60
    k_fold = 11
    x, y = read_data(dataset)
    results = pd.DataFrame()
    name = "vanilla RNN"
    results[name] = model_fit(x, y, neuron, n_epochs, n_batch, k_fold)
    name = "LSTM"
    results[name] = LSTM_model_fit(x, y, neuron, n_epochs, n_batch, k_fold)
    name = "GRU"
    results[name] = GRU_model_fit(x, y, neuron, n_epochs, n_batch, k_fold)
    
    print(results)
    print(results.describe())
    results.to_csv(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_DATA_COMPARISON_CATEGORICAL_NOES_SHUFFLE_K%s_E%s_B%s_N%s.csv' %(date, k_fold, n_epochs, n_batch, neuron))

    stop = timeit.default_timer()
    print ('Time :', stop-start)
    
    # save boxplot
    results.boxplot()
    plt.ylabel('accuracy')
    plt.xlabel('neurons')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_BOXPLOT_COMPARISON_CATEGORICAL_NOES_SHUFFLE_K%s_E%s_B%s_N%s.png' %(date, k_fold, n_epochs, n_batch, neuron))

experiment()