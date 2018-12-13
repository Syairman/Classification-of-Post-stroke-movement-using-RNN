import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import timeit

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from matplotlib import pyplot
from numpy.random import seed
seed(29)
from cntk.cntk_py import set_fixed_random_seed
set_fixed_random_seed(98)


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
    start = timeit.default_timer()
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_k_fold, shuffle=True, random_state=None)
    cvscores = []
    #es = EarlyStopping(monitor='loss', min_delta=10e-8 ,patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    for train, test in kfold.split(x,y):
        model = Sequential()
        model.add(SimpleRNN(n_neurons, activation='relu', input_shape=(x.shape[1], x.shape[2])))
        model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(x[train], y[train], epochs=n_epochs, batch_size=n_batch_size)
    

        # Evaluate the model
        scores = model.evaluate(x[test], y[test], verbose=0)
        print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    stop = timeit.default_timer()
    time = stop-start
    return cvscores, time

def experiment():
    date = "20181129"
    model = "RNN"
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-3\Data Merged\train_file.csv', header=None)
    neuron = 26
    n_epochs = 100
    n_batch = 60
    k_fold = [10, 11, 12, 13, 14]
    x, y = read_data(dataset)
    times = pd.DataFrame()
    results = pd.DataFrame()

    for k in k_fold:
        results[k], time = model_fit(x, y, neuron, n_epochs, n_batch, k)
        times.loc[0, k] = time
        results.to_csv(r'D:\Tugas Akhir\Final Dev-3\Results Data\%s_DATA_KFOLD_SHUFFLE_%s_K%s_E%s_B%s_N%s.csv' %(date, model, k, n_epochs, n_batch, neuron))
        results = pd.DataFrame()
    
    times.to_csv(r'D:\Tugas Akhir\Final Dev-3\Results Data\%s_DATA_TIME_KFOLD_SHUFFLE_%s_K%s_E%s_B%s_N%s.csv' %(date, model, k_fold, n_epochs, n_batch, neuron))

experiment()