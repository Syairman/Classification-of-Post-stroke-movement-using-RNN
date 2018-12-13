import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from numpy.random import seed
seed(29)
from cntk.cntk_py import set_fixed_random_seed
set_fixed_random_seed(98)

date = "20181031"

# Create Function to display confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(model, n_epochs, n_batch_size, n_neurons, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_CONFUSION_MATRIX_REGULARIZER_%s_%s_E%s_B%s_N%s.png' %(date, model, name, n_epochs, n_batch_size, n_neurons))

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

def model_fit(x, y, x_test, y_test, n_epochs, n_batch_size, n_neurons, model_arch, reg):
    es = EarlyStopping(monitor='val_loss', min_delta=0.005 ,patience=5, verbose=0, mode='auto')

    if model_arch == "RNN" :
        model = Sequential()
        model.add(SimpleRNN(n_neurons, activation='relu', input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=reg))
        model.add(Dense(12, activation="softmax", kernel_initializer='uniform'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
        
        #Fit the model
        history = model.fit(x, y, epochs=n_epochs, validation_data=(x_test, y_test), batch_size=n_batch_size, callbacks=[es])


    elif model_arch == "LSTM":
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=reg))
        model.add(Dense(12, activation="softmax", kernel_initializer="uniform"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy, metrics.binary_accuracy])
        
        #Fit the model
        history = model.fit(x, y, epochs=n_epochs, validation_data=(x_test, y_test), batch_size=n_batch_size, callbacks=[es])
    elif model_arch == "GRU":
        model = Sequential()
        model.add(GRU(n_neurons, input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=reg))
        model.add(Dense(12, activation="softmax", kernel_initializer="uniform"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
        
        #Fit the model
        history = model.fit(x, y, epochs=n_epochs, validation_data=(x_test, y_test), batch_size=n_batch_size, callbacks=[es])
    else :
        print ("Error Model Input")
        exit()
    
    model.save(r'D:\Tugas Akhir\Final Dev-4\Model h5\%s_%s_REGULARIZER_%s_E%s_B%s_N%s.h5' %(date, model_arch, name, n_epochs, n_batch_size, n_neurons))
    
    data = pd.DataFrame()
    data['loss'] = history.history['loss']
    data['categorical_accuracy'] = history.history['categorical_accuracy']
    data['val_loss'] = history.history['val_loss']
    data['val_categorical_accuracy'] = history.history['val_categorical_accuracy']
    data.to_csv(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_DATA_FINAL_REGULARIZER_%s_%s_E%s_B%s_N%s.csv' %(date, model_arch, name, n_epochs, n_batch_size, n_neurons))
    
    #Sumarize history for accuracy
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_ACC_FINAL_REGULARIZER_%s_%s_E%s_B%s_N%s.png' %(date, model_arch, name, n_epochs, n_batch_size, n_neurons))

    #Summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(r'D:\Tugas Akhir\Final Dev-4\Results Data\%s_LOSS_FINAL_REGULARIZER_%s_%s_E%s_B%s_N%s.png' %(date, model_arch, name, n_epochs, n_batch_size, n_neurons))

    #Predict Value
    y_test_pred = model.predict(x_test)

    #Plot Confusion Matrix
    cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))
    np.set_printoptions(precision=2)

    class_names = np.array(['Jari Abduksi','Jari Fleksi','Jari Hiperekstensi','Jempol Abduksi', 'Jempol Fleksi', 'Jempol Hiperekstensi', 'Lengan Pronasi', 'Lengan Supinasi', 'Pergelangan Abduksi','Pergelangan Adduksi', 'Pergelangan Fleksi', 'Pergelangan Hiperekstensi'])
    plot_confusion_matrix(model_arch, n_epochs, n_batch_size, n_neurons, cnf_matrix, classes=class_names, normalize=False, title='Confusion Matrix Without normalization')
    

def experiment():
    global date
    date = "20181204"
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    datates = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\test_file.csv', header=None)
    x, y = read_data(dataset)
    x_t, y_t = read_data(datates)
    epochs = 100
    batch_size = 60
    neuron = 23
    model = "RNN"
    regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.001, l2=0.0), L1L2(l1=0.0, l2=0.001), L1L2(l1=0.001, l2=0.001)]
    for reg in regularizer:
        global name
        name = ("l1 %.3f l2 %.3f" %(reg.l1, reg.l2))
        model_fit(x, y, x_t, y_t, epochs, batch_size, neuron, model, reg)

def experiment_1():
    global date
    date = "20181210"
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    datates = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\test_file.csv', header=None)
    x, y = read_data(dataset)
    x_t, y_t = read_data(datates)
    epochs = 100
    batch_size = 60
    neuron = 21
    model = "LSTM"
    regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.001, l2=0.0), L1L2(l1=0.0, l2=0.001), L1L2(l1=0.001, l2=0.001)]
    for reg in regularizer:
        global name
        name = ("l1 %.3f l2 %.3f" %(reg.l1, reg.l2))
        model_fit(x, y, x_t, y_t, epochs, batch_size, neuron, model, reg)

def experiment_2():
    global date
    date = "20181204"
    dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\train_file.csv', header=None)
    datates = pd.read_csv(r'D:\Tugas Akhir\Final Dev-4\Data Merged\test_file.csv', header=None)
    x, y = read_data(dataset)
    x_t, y_t = read_data(datates)
    epochs = 100
    batch_size = 60
    neuron = 23
    model = "GRU"
    regularizer = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.001, l2=0.0), L1L2(l1=0.0, l2=0.001), L1L2(l1=0.001, l2=0.001)]
    for reg in regularizer:
        global name
        name = ("l1 %.3f l2 %.3f" %(reg.l1, reg.l2))
        model_fit(x, y, x_t, y_t, epochs, batch_size, neuron, model, reg)


experiment_1()

