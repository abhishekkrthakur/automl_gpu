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

dataset = "helena"

train_data = cPickle.load(open(dataset + '_train.pkl', 'rb'))
test_data = cPickle.load(open(dataset + '_test.pkl', 'rb'))
valid_data = cPickle.load(open(dataset + '_valid.pkl', 'rb'))
labels = cPickle.load(open(dataset + '_labels.pkl', 'rb'))

svd = preprocessing.StandardScaler()
train_data = svd.fit_transform(train_data)
valid_data = svd.transform(valid_data)
test_data = svd.transform(test_data)

test_preds = []
valid_preds = []
NUM_ROUND = 4
for i in range(NUM_ROUND):
    print "=============", i
    dims = train_data.shape[1]

    model = Sequential()
    model.add(Dense(400, input_shape=(dims,)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Dense(labels.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")

    model.fit(train_data, labels, nb_epoch=200, batch_size=128)
    tp = model.predict(test_data)
    yp = model.predict(valid_data)
    if i == 0:
        test_preds = tp
        valid_preds = yp
    else:
        test_preds += tp
        valid_preds += yp

test_preds = test_preds * 1./NUM_ROUND
valid_preds = valid_preds * 1./NUM_ROUND

np.savetxt('res/' + dataset + '_test_001.predict', test_preds, '%1.10f')
np.savetxt('res/' + dataset + '_valid_001.predict', valid_preds, '%1.10f')

