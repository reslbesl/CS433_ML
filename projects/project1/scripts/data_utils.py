# -*- coding: utf-8 -*-
""" Helper functions for data splitting and test data generation"""

import numpy as np

from os import path

from proj1_helpers import *

SEED = 42

FEATURE_NAMES = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
                 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
                 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta',
                 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
                 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

JET_NUM_IDX = FEATURE_NAMES.index('PRI_jet_num')


def train_eval_split(y, tx, split_ratio, seed=SEED):
    """
    Split a dataset into train and evaluation sets based on the split ratio.

    If split_ratio is 0.8 you will have 80% of your data set dedicated to training
    and the rest dedicated to testing

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param split_ratio: float: scalar in (0, 1] that defines what fraction of records will be assigned to the training set
    :param seed: int: seed for pseudo-random number generator

    :return (tx_train, y_train): tuple: training data
    :return (tx_test, y_test): tuple: test data
    """
    # set seed for reproducibility
    np.random.seed(seed)

    num_samples = len(y)
    num_train = int(np.ceil(num_samples * split_ratio))

    shuffle_indices = np.random.permutation(num_samples)
    shuffled_y = y[shuffle_indices]
    shuffled_x = tx[shuffle_indices]

    return (shuffled_x[:num_train], shuffled_y[:num_train]), (shuffled_x[num_train:], shuffled_y[num_train:])


def k_fold_iter(y, tx, k_fold, seed=SEED):
    """
    Generate a k-fold iterator for dataset (tx, y)

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives a k-fold train-test split of the data.

    Example use:
    for (x_train y_train), (x_test, y_test) in k_fold_iter(y, tx, 4):
        <DO_SOMETHING>

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param k_fold: int: number of dataset folds to be returned
    :param seed: int: seed for pseudo-random number generator
    """
    np.random.seed(seed)

    num_samples = len(y)
    interval = int(num_samples / k_fold)

    indices = np.random.permutation(num_samples)

    for k in range(k_fold):
        test_idx = indices[k * interval: (k + 1) * interval]
        train_idx = np.ones(num_samples).astype(bool)
        train_idx[test_idx] = False

        yield (tx[train_idx], y[train_idx]), (tx[test_idx], y[test_idx])


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=SEED):
    """
    Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param batch_size: int: size of each batch
    :param: num_batches: int: number of batches
    :param shuffle: bool: Whether to shuffle data to avoid ordering in original data
    :param seed: int: seed for pseudo-random number generator
    """
    # set seed for reproducibility
    np.random.seed(seed)

    num_samples = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, num_samples)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index], batch_num
            
def load_data_split_categorical(data_path, cat):
    '''
    Splits data according to the categorical feature "PRI_jet_num" provided as 'cat' argument (cat in range (0,4) ). Performs also some feature removal according
    to the dataset documentation, based on the value of cat. 
    '''
    #LOAD TRAIN DATA
    s = np.genfromtxt(path.join(data_path,'train.csv'), delimiter=",", skip_header=1)
    y = np.genfromtxt(path.join(data_path,'train.csv'), delimiter=",", skip_header=1, dtype=str, usecols=1)
    
    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = LABELS['b']
    
    #insert new column with binary label in s
    s = np.delete(s,1,axis=1)
    s = np.insert(s,1,yb,axis=1)
    
    #split data set according to the value of PRI_JET_NUM
    sc = s[np.where(s[:,24]==cat)]
    y = sc[:,1].astype(np.int)
    ids = sc[:,0].astype(np.int)
    x = sc[:, 2:]

    #a bit of feature selection based on cat. Basically removing undefined columns
    if cat <= 1:
        if cat == 0:
            undef_c = [4,5,6,12,22,23,24,25,26,27,28,29]
        else:
            undef_c = [4,5,6,12,22,26,27,28]
    if cat > 1:
            undef_c = [22]    
    x = np.delete(x,undef_c,axis=1)
    
    #Normalise data
    x, mean_x, std_x = standardise(x)

    # Split into train and evaluation set
    (x_train, y_train), (x_eval, y_eval) = train_eval_split(y, x, split_ratio=.7, seed=SEED)
    tx_train = np.c_[np.ones(len(y_train)), x_train]
    tx_eval = np.c_[np.ones(len(y_eval)), x_eval]

    #LOAD TEST DATA
    s = np.genfromtxt(path.join(data_path,'test.csv'), delimiter=",", skip_header=1)
    y = np.genfromtxt(path.join(data_path,'test.csv'), delimiter=",", skip_header=1, dtype=str, usecols=1)
    
    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = LABELS['b']
    
    #insert new column with binary label in s
    s = np.delete(s,1,axis=1)
    s = np.insert(s,1,yb,axis=1)
    
    #split data set according to the value of PRI_JET_NUM
    sc = s[np.where(s[:,24]==cat)]

    y_test = sc[:,1].astype(np.int)
    ids_test = sc[:,0].astype(np.int)
    x_test = sc[:, 2:]

    #a bit of feature selection based on cat. Basically removing undefined columns
    if cat <= 1:
        if cat == 0:
            undef_c = [4,5,6,12,22,23,24,25,26,27,28,29]
        else:
            undef_c = [4,5,6,12,22,26,27,28]
    if cat > 1:
            undef_c = [22]    
    x_test = np.delete(x_test,undef_c,axis=1)

    # Don't forget to standardise to same mean and std
    x_test = standardise_to_fixed(x_test, mean_x, std_x)
    tx_test = np.c_[np.ones(len(y_test)), x_test]

    return (y_train, x_train, tx_train), (y_eval, x_eval, tx_eval), (y_test, x_test, tx_test, ids_test)


def generate_mask(features_to_remove):
    feat_idx = [FEATURE_NAMES.index(f) for f in features_to_remove]
    mask = np.ones(len(FEATURE_NAMES)).astype(bool)
    mask[feat_idx] = False

    return mask


def feature_transform(x):
    x_ind = (x[:, JET_NUM_IDX] > 1).astype(int)
    x = np.concatenate([x, np.expand_dims(x_ind, axis=1)], axis=1)

    features = generate_mask(['DER_deltaeta_jet_jet', 'DER_prodeta_jet_jet'])
    features = np.concatenate([features, [True]])

    x = x[:, features]

    return x

     
