import numpy as np

from os import path

from utils import load_csv_data, eval_model, predict_labels, create_csv_submission
from data_utils import standardise, standardise_to_fixed, feature_transform_polybasis
from implementation_variants import logistic_regression_mean

cwd = path.dirname(__file__)

SEED = 42
DATA_PATH = path.join(cwd, '../data/')

# Training hyperparameters (obtained through procedure in Run.ipynb)
MAX_ITERS = 50000
GAMMA = 0.01
THRESHOLD = 1e-7

if __name__ == "__main__":
    print('Load train data...')
    # Load train data
    y_train, x_train, _ = load_csv_data(path.join(DATA_PATH, 'train.csv'))

    print('Apply feature transform...')
    # Apply feature transform
    fx_train = feature_transform_polybasis(x_train)

    # Standardise to mean and s.d.
    fx_train, mu_train, sigma_train = standardise(fx_train)

    # Add offset term
    tx_train = np.c_[np.ones(len(y_train)), fx_train]

    # Initialise training
    w_initial = np.zeros(tx_train.shape[1])

    print('Run LogReg gradient descent..')
    # Run gradient descent
    w, loss = logistic_regression_mean(y_train, tx_train, w_initial, MAX_ITERS, GAMMA, verbose=True)
    print(f'Training loss: {loss}')

    acc = eval_model(y_train, tx_train, w, thresh=0.5)
    print(f'Training accuracy: {acc}')

    print('Load test data...')
    # Load test data
    y_test, x_test, ids_test = load_csv_data(path.join(DATA_PATH, 'test.csv'))
    fx_test = feature_transform_polybasis(x_test)

    # Standardise to mean and s.d. of training data
    fx_test = standardise_to_fixed(fx_test, mu_train, sigma_train)

    # Add offset term
    tx_test = np.c_[np.zeros(fx_test.shape[0]), fx_test]

    # Get predictions on test set
    y_pred = predict_labels(w, tx_test, thresh=0.5)
    create_csv_submission(ids_test, y_pred, 'final_submission.csv')



