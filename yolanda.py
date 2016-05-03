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
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

dataset = "yolanda"

train_data = cPickle.load(open(dataset + '_train.pkl', 'rb'))
test_data = cPickle.load(open(dataset + '_test.pkl', 'rb'))
valid_data = cPickle.load(open(dataset + '_valid.pkl', 'rb'))
labels = cPickle.load(open(dataset + '_labels.pkl', 'rb'))


svd = preprocessing.StandardScaler()
train_data = svd.fit_transform(train_data)
valid_data = svd.transform(valid_data)
test_data = svd.transform(test_data)


NUM_ROUNDS = 5

test_preds = np.zeros((test_data.shape[0], NUM_ROUNDS))
valid_preds = np.zeros((valid_data.shape[0], NUM_ROUNDS))
for i in range(NUM_ROUNDS):
    print "=============", i

    dims = train_data.shape[1]

    model = Sequential()
    model.add(Dense(150, init='glorot_uniform', input_shape=(dims,)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.1))

    model.add(Dense(150))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_data, labels*1./1000., nb_epoch=20, batch_size=128)
    tp = model.predict(test_data)
    yp = model.predict(valid_data)
    test_preds[:,i] = tp.ravel() * 1000.
    valid_preds[:,i] = yp.ravel() * 1000.


test_preds = np.mean(test_preds, axis=1)
valid_preds = np.mean(valid_preds, axis=1)

np.savetxt('res/' + dataset + '_test_001.predict', test_preds, '%1.10f')
np.savetxt('res/' + dataset + '_valid_001.predict', valid_preds, '%1.10f')
