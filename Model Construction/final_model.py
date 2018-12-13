import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from keras.models import Sequential
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

# Create Function to display confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    plt.savefig(r'D:\Tugas Akhir\Final Dev\12_Classifier\Results Data\Confusion Matrix\PLOT\2018109_LSTM_NoCB_E%s_B%s_N%s_L1%s_L2%s.png' %(n_epochs, n_batch_size, n_neurons, l_r1, l_r2))
    plt.show()

dataset = pd.read_csv(r'D:\Tugas Akhir\Final Dev\12_Classifier\Data Merged\train_file.csv', header=None)
datates = pd.read_csv(r'D:\Tugas Akhir\Final Dev\12_Classifier\Data Merged\test_file.csv', header=None)


x = dataset.iloc[:, :26].values
y = dataset.iloc[:, 26].values
x_test = datates.iloc[:, :26].values
y_test = datates.iloc[:, 26].values

t = 571
x = np.reshape(x, (int(x.shape[0]/t), t, x.shape[1]))
x_test = np.reshape(x_test, (int(x_test.shape[0]/t), t, x_test.shape[1]))

indices = [ind for ind in range(len(y)) if ind%t==0]
y = y[indices]
y_true = y
indices_test = [ind for ind in range(len(y_test)) if ind%t==0]
y_test = y_test[indices_test]

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_new = label_encoder.fit_transform(y)
y = y_new
y_new_test = label_encoder.fit_transform(y_test)
y_test = y_new_test

from sklearn.preprocessing import LabelBinarizer
multi_label = LabelBinarizer()
y_new = multi_label.fit_transform(y)
y = y_new
y_new_test = multi_label.fit_transform(y_test)
y_test = y_new_test
y_test_true = y_test



n_neurons = 26
n_epochs = 100
n_batch_size = 60
l_r1 = 0
l_r2 = 0
reg = L1L2(l1=l_r1, l2 =l_r2)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)


model = Sequential()
model.add(LSTM(n_neurons, input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=reg))
model.add(Dense(12, activation="softmax", kernel_initializer="uniform"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
#Fit the model
history = model.fit(x, y, epochs=n_epochs, validation_data=(x_test, y_test), batch_size=n_batch_size)

print (history.history.keys())
#Sumarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(r'D:\Tugas Akhir\Final Dev\12_Classifier\Results Data\Acc and Loss Graph\2018109_ACC_LSTM_NoCB_E%s_B%s_N%s_L1%s_L2%s.png' %(n_epochs, n_batch_size, n_neurons, l_r1, l_r2))
plt.show()

#Summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(r'D:\Tugas Akhir\Final Dev\12_Classifier\Results Data\Acc and Loss Graph\2018109_LOSS_LSTM_NoCB_E%s_B%s_N%s_L1%s_L2%s.png' %(n_epochs, n_batch_size, n_neurons, l_r1, l_r2))
plt.show()


# Evaluate the model
scores = model.evaluate(x_test, y_test)
print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save(r'D:\Tugas Akhir\Final Dev\12_Classifier\Model h5\2018109_LSTM_NoCB_E%s_B%s_N%s_L1%s_L2%s_LAPTOP.h5' %(n_epochs, n_batch_size, n_neurons, l_r1, l_r2))

y_test_pred = model.predict(x_test)

# Plot Confusion Matrix
cnf_matrix = confusion_matrix(y_test_true.argmax(axis=1), y_test_pred.argmax(axis=1))
np.set_printoptions(precision=2)
## Plot non Normalized Confusion Matrix
class_names = np.array(['Jari Abduksi','Jari Fleksi','Jari Hiperekstensi','Jempol Abduksi', 'Jempol Fleksi', 'Jempol Hiperekstensi', 'Lengan Pronasi', 'Lengan Supinasi', 'Pergelangan Abduksi','Pergelangan Adduksi', 'Pergelangan Fleksi', 'Pergelangan Hiperekstensi'])
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion Matrix Without normalization')