# coding: utf-8
"""
AutoML GPU Track
__author__ : Abhishek Thakur
"""

import cPickle
import numpy as np
import sys
sys.setrecursionlimit(100000)
from sklearn import preprocessing
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

dataset = "evita"

train_data = cPickle.load(open(dataset + '_train.pkl', 'rb')).toarray()
test_data = cPickle.load(open(dataset + '_test.pkl', 'rb')).toarray()
valid_data = cPickle.load(open(dataset + '_valid.pkl', 'rb')).toarray()
labels = cPickle.load(open(dataset + '_labels.pkl', 'rb'))


train_data = np.column_stack((train_data, 
                              np.sum(train_data, axis=1), 
                             train_data.shape[1] - np.sum(train_data, axis=1)))

valid_data = np.column_stack((valid_data, np.sum(valid_data, axis=1),
                             valid_data.shape[1] - np.sum(valid_data, axis=1)))

test_data = np.column_stack((test_data, 
                             np.sum(test_data, axis=1),
                            test_data.shape[1] - np.sum(test_data, axis=1)))

svd = preprocessing.MinMaxScaler()
train_data = svd.fit_transform(train_data)
valid_data = svd.transform(valid_data)
test_data = svd.transform(test_data)

dims = train_data.shape[1]

y_c = np_utils.to_categorical(labels)
nb_classes = y_c.shape[1]

NUM_ROUNDS = 10
test_preds = np.zeros((test_data.shape[0], NUM_ROUNDS))
valid_preds = np.zeros((valid_data.shape[0], NUM_ROUNDS))
for i in range(NUM_ROUNDS):
    print "=============", i
    
    dims = train_data.shape[1]

    y_c = np_utils.to_categorical(labels)
    nb_classes = y_c.shape[1]

    model = Sequential()
    model.add(Dense(1500, input_shape=(dims,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(PReLU())

    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(PReLU())

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer="adam")

    model.fit(train_data, y_c, nb_epoch=15, batch_size=128)
    tp = model.predict_proba(test_data)[:, 1]
    yp = model.predict_proba(valid_data)[:, 1]
    test_preds[:, i] = tp
    valid_preds[:, i] = yp

test_preds = np.mean(test_preds, axis=1)
valid_preds = np.mean(valid_preds, axis=1)

np.savetxt('res/' + dataset + '_test_001.predict', test_preds, '%1.10f')
np.savetxt('res/' + dataset + '_valid_001.predict', valid_preds, '%1.10f')


