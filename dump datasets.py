# coding: utf-8
"""
AutoML GPU Track
This module dumps datasets as pickles
__author__ : Abhishek Thakur
"""

from data_io import *
import cPickle

train_data = data_binary_sparse('evita/evita_train.data', nbr_features=3000)
test_data = data_binary_sparse('evita/evita_test.data', nbr_features=3000)
valid_data = data_binary_sparse('evita/evita_valid.data', nbr_features=3000)
feat_type = np.loadtxt('evita/evita_feat.type', dtype = 'S20')
labels = np.loadtxt('evita/evita_train.solution')

cPickle.dump(train_data, open('evita_train.pkl', 'wb'), -1)
cPickle.dump(test_data, open('evita_test.pkl', 'wb'), -1)
cPickle.dump(valid_data, open('evita_valid.pkl', 'wb'), -1)
cPickle.dump(labels, open('evita_labels.pkl', 'wb'), -1)

train_data = data_sparse('flora/flora_train.data', nbr_features=200000)
test_data = data_sparse('flora/flora_test.data', nbr_features=200000)
valid_data = data_sparse('flora/flora_valid.data', nbr_features=200000)
feat_type = np.loadtxt('flora/flora_feat.type', dtype = 'S20')
labels = np.loadtxt('flora/flora_train.solution')

cPickle.dump(train_data, open('flora_train.pkl', 'wb'), -1)
cPickle.dump(test_data, open('flora_test.pkl', 'wb'), -1)
cPickle.dump(valid_data, open('flora_valid.pkl', 'wb'), -1)
cPickle.dump(labels, open('flora_labels.pkl', 'wb'), -1)

train_data = np.loadtxt('helena/helena_train.data')
test_data = np.loadtxt('helena/helena_test.data')
valid_data = np.loadtxt('helena/helena_valid.data')
feat_type = np.loadtxt('helena/helena_feat.type', dtype = 'S20')
labels = np.loadtxt('helena/helena_train.solution')

cPickle.dump(train_data, open('helena_train.pkl', 'wb'), -1)
cPickle.dump(test_data, open('helena_test.pkl', 'wb'), -1)
cPickle.dump(valid_data, open('helena_valid.pkl', 'wb'), -1)
cPickle.dump(labels, open('helena_labels.pkl', 'wb'), -1)

train_data = data_sparse('tania/tania_train.data', nbr_features=47236)
test_data = data_sparse('tania/tania_test.data', nbr_features=47236)
valid_data = data_sparse('tania/tania_valid.data', nbr_features=47236)
feat_type = np.loadtxt('tania/tania_feat.type', dtype = 'S20')
labels = np.loadtxt('tania/tania_train.solution')

cPickle.dump(train_data, open('tania_train.pkl', 'wb'), -1)
cPickle.dump(test_data, open('tania_test.pkl', 'wb'), -1)
cPickle.dump(valid_data, open('tania_valid.pkl', 'wb'), -1)
cPickle.dump(labels, open('tania_labels.pkl', 'wb'), -1)

train_data = np.loadtxt('yolanda/yolanda_train.data')
test_data = np.loadtxt('yolanda/yolanda_test.data')
valid_data = np.loadtxt('yolanda/yolanda_valid.data')
feat_type = np.loadtxt('yolanda/yolanda_feat.type', dtype = 'S20')
labels = np.loadtxt('yolanda/yolanda_train.solution')

cPickle.dump(train_data, open('yolanda_train.pkl', 'wb'), -1)
cPickle.dump(test_data, open('yolanda_test.pkl', 'wb'), -1)
cPickle.dump(valid_data, open('yolanda_valid.pkl', 'wb'), -1)
cPickle.dump(labels, open('yolanda_labels.pkl', 'wb'), -1)



