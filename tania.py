# coding: utf-8
"""
AutoML GPU Track
__author__ : Abhishek Thakur
"""

from sklearn import decomposition

import cPickle
import numpy as np
import sys
sys.setrecursionlimit(100000)

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


dataset = "tania"
train_data = cPickle.load(open(dataset + '_train.pkl', 'rb'))
test_data = cPickle.load(open(dataset + '_test.pkl', 'rb'))
valid_data = cPickle.load(open(dataset + '_valid.pkl', 'rb'))
labels = cPickle.load(open(dataset + '_labels.pkl', 'rb'))

svd = decomposition.TruncatedSVD(n_components=200)
train_data = svd.fit_transform(train_data)
valid_data = svd.transform(valid_data)
test_data = svd.transform(test_data)

test_preds = []
valid_preds = []
for i in range(5):
    print "=============", i
    dims = train_data.shape[1]

    model = Sequential()
    model.add(Dense(900, input_shape=(dims,)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Dense(900))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.3))

    model.add(Dense(labels.shape[1]))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")
    model.fit(train_data, labels, nb_epoch=20, batch_size=128)
    tp = model.predict(test_data)
    yp = model.predict(valid_data)
    if i == 0:
        test_preds = tp
        valid_preds = yp
    else: 
        test_preds += tp
        valid_preds += yp

test_preds = test_preds * 1./5.
valid_preds = valid_preds * 1./5.

np.savetxt('res/' + dataset + '_test_001.predict', test_preds, '%1.5f')
np.savetxt('res/' + dataset + '_valid_001.predict', valid_preds, '%1.5f')
