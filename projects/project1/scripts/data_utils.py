# -*- coding: utf-8 -*-
""" Helper functions for data splitting and test data generation"""

import numpy as np

SEED = 42

FEATURE_NAMES = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
                 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
                 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta',
                 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
                 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

LEAST_INFO = ['PRI_lep_pt', 'PRI_met', 'DER_pt_tot', 'DER_mass_vis', 'DER_deltar_tau_lep',
              'PRI_met_phi', 'PRI_tau_phi', 'PRI_lep_phi', 'PRI_lep_eta', 'PRI_tau_eta']

JET_NOT_DEFINED = ['DER_lep_eta_centrality', 'DER_prodeta_jet_jet', 'DER_mass_jet_jet', 'DER_deltaeta_jet_jet',
                   'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_subleading_pt']

JET_NUM_IDX = FEATURE_NAMES.index('PRI_jet_num')


def standardise(x):
    """ Standardise array x where rows are samples and columns contain features """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    std_x = (x - mu) / sigma

    return std_x, mu, sigma


def standardise_to_fixed(x, mu, sigma):
    """ Standardise array x to given mean and standard deviation """

    return (x - mu) / sigma


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    phi = np.hstack([x**i for i in range(1, degree+1)])

    return phi


def generate_mask(features_to_remove):
    """Generate a boolean mask to select only specific columns from data matrix"""
    feat_idx = [FEATURE_NAMES.index(f) for f in features_to_remove]
    mask = np.ones(len(FEATURE_NAMES)).astype(bool)
    mask[feat_idx] = False

    return mask


def generate_feature_idx(feature_to_idx):
    """Get column indices for a list of feature names"""
    return [FEATURE_NAMES.index(f) for f in feature_to_idx]


def feature_transform_mostinfo(x):
    # Generate feature mask
    feature_mask = generate_mask(LEAST_INFO)

    # Remove unwanted features
    x = x[:, feature_mask]

    return x


def feature_transform_imputejet(x):
    feature_idx = generate_feature_idx(JET_NOT_DEFINED)
    for i in feature_idx:
        impute_val = np.mean(x[:, i][x[:, i] != -999.])
        x[:,i][x[:, i] == -999.] = impute_val

    return x


def feature_transform_polybasis(x, degree=4):

    fx_list = []

    # Expand polynomial basis for DEFINED values
    for i in range(x.shape[1]):

        # Build poly basis
        fx_poly = build_poly(x[:, i], degree)

        # Replace undefined value with same value for all polys
        fx_poly[x[:, i] == -999.] = -999

        fx_list.append(fx_poly)

    fx = np.hstack(fx_list)

    return fx
