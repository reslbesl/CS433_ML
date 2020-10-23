# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

from os import path

from data_utils import train_eval_split

SEED = 42
LABELS = {'b': 0, 's': 1}
PRI_JET_NUM_IDX = 22


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), input_data (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = LABELS['b']
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, thresh=0):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= thresh)] = LABELS['b']
    y_pred[np.where(y_pred > thresh)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to AICrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    # Re-code to match labels on submission platform

    y_pred[y_pred == LABELS['b']] = -1
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def load_data(data_path):
    # Load train data
    y, x, ids = load_csv_data(path.join(data_path, 'train.csv'))

    # Split into train and evaluation set
    (x_train, y_train), (x_eval, y_eval) = train_eval_split(y, x, split_ratio=.7, seed=SEED)

    # Load test data
    y_test, x_test, ids_test = load_csv_data(path.join(data_path, 'test.csv'))

    # Standardise to training set mean
    x_train, mu, sigma = standardise(x_train)
    x_eval = standardise_to_fixed(x_eval, mu, sigma)
    x_test = standardise_to_fixed(x_test, mu, sigma)

    # Generate x tilde
    tx_train = np.c_[np.ones(len(y_train)), x_train]
    tx_eval = np.c_[np.ones(len(y_eval)), x_eval]
    tx_test = np.c_[np.ones(len(y_test)), x_test]

    return (y_train, x_train, tx_train), (y_eval, x_eval, tx_eval), (y_test, x_test, tx_test, ids_test)


def standardise(x):
    """ Standardise array x where rows are samples and columns contain features """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    std_x = (x - mu) / sigma

    return std_x, mu, sigma


def standardise_to_fixed(x, mu, sigma):
    """ Standardise array x to given mean and standard deviation """

    return (x - mu) / sigma


def get_accuracy(y_pred, y_true):
    """ Calculate accuracy of predictions y_pred in predicting true labels y_true """

    return sum(y_pred == y_true)/len(y_true)


def get_pos_rates(y_pred, y_true):
    """ Calculate true and false positive rates of predicted labels y_pred in predicting true labels y_true """
    tp = sum(y_pred[y_true == 1] == 1)
    fp = sum(y_true == 1) - tp

    return tp, fp


def get_neg_rates(y_pred, y_true):
    """ Calculate true and false negative rates of predicted labels y_pred in predicting true labels y_true """
    tn = sum(y_pred[y_true == -1] == -1)
    fn = sum(y_true == -1) - tn

    return tn, fn


def get_f1_score(y_pred, y_true):
    """ Calculate F1 score of predicted labels y_pred in predicting true labels y_true """
    tp, fp = get_pos_rates(y_pred, y_true)
    tn, fn = get_neg_rates(y_pred, y_true)

    return tp/(tp + (fp + fn)/2.)


def eval_model(y, x, w, thresh=0):
    y_pred = predict_labels(w, x, thresh)
    acc = get_accuracy(y_pred, y)

    return acc
