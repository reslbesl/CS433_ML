import numpy as np

# Main functions to implement
def least_squares(y, tx):
    """
    Linear regression using normal equations

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    """
    # Compute Gram Matrix
    gram = tx.T.dot(tx)

    # Solve the linear system from normal equations
    w = np.linalg.solve(gram, tx.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Normal equations using L2 regularization

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param lambda_: float: penalty parameter
    """
    assert lambda_ > 0, "Penalty factor must be positive."

    if len(tx.shape) > 1:
        num_samples, num_dims = tx.shape
    else:
        num_samples, num_dims = len(y), 1

    # Compute Gram matrix
    gram = tx.T.dot(tx)

    # Compute identity dxd matrix
    eye = np.identity(num_dims)

    # Compute lambda prime as lamda*2N
    plambda = lambda_ * 2 * num_samples

    # Solve the linear system from normal equation under L2 regularization
    w = np.linalg.solve((gram + plambda * eye), tx.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial modeo parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_mse(y, tx, w)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):

            # Compute gradient for current batch
            grad = compute_gradient_mse(batch_y, batch_tx, w)

            # Update model parameters
            w = w - gamma * grad

            # Compute new loss
            loss = compute_loss_mse(y, tx, w)

            print("Stochastic GD({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss


# Variants
def lasso_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Lasso regression using subgradient method

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_lasso(y, tx, w, lambda_)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss




def least_squares_SGD_robbinson(y, tx, initial_w, max_iters, r_gamma=.7):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1 and decreasing step-size.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert .5 < r_gamma < 1, 'Parameter r must be in (0.5, 1)'

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):

            # Compute gradient for current batch
            grad = compute_gradient_mse(batch_y, batch_tx, w)

            # Update step size
            gamma = 1/((n_iter + 1)**r_gamma)

            # Update model parameters
            w = w - gamma * grad

            # Compute new loss
            loss = compute_loss_mse(y, tx, w)

            print("Stochastic GD({bi}/{ti}): loss={l}, gradient={grad}, gamma={gam}".format(bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad), gam=gamma))

    return w, loss


# Dependencies
def compute_loss_mse(y, tx, w):
    """ Calculate the MSE loss under weights vector w. """
    e = y - tx.dot(w)

    return e.dot(e)/(2*len(e))


def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    e = y - tx.dot(w)

    return -tx.T.dot(e) / len(e)


def compute_loss_lasso(y, tx, w, lambda_):
    """ Calculate the Lasso loss under weights vector w. """
    e = y - tx.dot(w)

    return e.dot(e)/(2 * len(e)) + lambda_ * sum(abs(w))


def compute_gradient_lasso(y, tx, w, lambda_):
    e = y - tx.dot(w)
    subgrad = lambda_ * np.sign(w)

    return -tx.T.dot(e)/len(e) + subgrad


# Helper functions
def standardise(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    std_x = (x - mu) / sigma

    return std_x, mu, sigma


def standardise_to_fixed(x, mu, sigma):

    return (x - mu) / sigma


def train_eval_split(y, tx, split_ratio, seed=42):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed for reproducibility
    np.random.seed(seed)

    num_samples = len(y)
    num_train = int(np.ceil(num_samples * split_ratio))

    shuffle_indices = np.random.permutation(num_samples)
    shuffled_y = y[shuffle_indices]
    shuffled_x = tx[shuffle_indices]

    return (shuffled_x[:num_train], shuffled_y[:num_train]), (shuffled_x[num_train:], shuffled_y[num_train:])


def k_fold_iter(y, tx, k_fold, seed=42):
    """
    Generate a k-fold iterator for dataset (tx, y)

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives a k-fold train-test split of the data.

    Example use:
    for (x_train y_train), (x_test, y_test) in k_fold_iter(y, tx, 4):
        <DO_SOMETHING>

    :param y:
    :param tx:
    :param k_fold:
    :param seed:
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


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=42):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def get_accuracy(y_pred, y_true):
    return sum(y_pred == y_true)/len(y_true)


def get_pos_rates(y_pred, y_true):
    tp = sum(y_pred[y_true == 1] == 1)
    fp = sum(y_true == 1) - tp

    return tp, fp


def get_neg_rates(y_pred, y_true):
    tn = sum(y_pred[y_true == -1] == -1)
    fn = sum(y_true == -1) - tn

    return tn, fn


def get_f1_score(y_pred, y_true):
    tp, fp = get_pos_rates(y_pred, y_true)
    tn, fn = get_neg_rates(y_pred, y_true)

    return tp/(tp + (fp + fn)/2.)
