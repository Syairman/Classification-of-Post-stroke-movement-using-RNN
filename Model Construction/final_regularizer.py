import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import timeit

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.utils import np_utils
from keras.regularizers import L1L2
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

def model_fit(x, y, n_neurons, n_epochs, n_batch_size, n_k_fold, reg):

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_k_fold, shuffle=False, random_state=None)
    cvscores = []
    from keras.initializers import glorot_uniform


    for train, test in kfold.split(x,y):
        model = Sequential()
        model.add(SimpleRNN(n_neurons, activation='relu', input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=reg))
        model.add(Dense(12, activation="softmax", kernel_initializer=glorot_uniform(seed=set_fixed_random_seed(29))))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(x[train], y[train], epochs=n_epochs, batch_size=n_batch_size)

        # Evaluate the model
        scores = model.evaluate(x[test], y[test], verbose=0)
        print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    return cvscores

def experiment():
    dataset = pd.read_csv(r'G:\Final Dev\12_Classifier\Data Merged\train_file.csv', header=None)
    neuron = 26
    n_epochs = 5
    n_batch = 144
    k_fold = 3
    x, y = read_data(dataset)
    results = pd.DataFrame()
    regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.001, l2=0.0), L1L2(l1=0.0, l2=0.001), L1L2(l1=0.001, l2=0.001)]
    for reg in regularizer:
        name = ("l1 %.3f l2 %.3f" %(reg.l1, reg.l2))
        results[name] = model_fit(x, y, neuron, n_epochs, n_batch, k_fold, reg)
    print(results)
    print(results.describe())

    stop = timeit.default_timer()
    print ('Time :', stop-start)

    # save boxplot
    results.boxplot()
    plt.ylabel('Accuracy')
    plt.xlabel('Regularizer')
    Box_Name = input("Nama Figure: ")
    plt.savefig(r'G:\Final Dev\12_Classifier\Box Plot Model\%s.png' %Box_Name)

experiment()