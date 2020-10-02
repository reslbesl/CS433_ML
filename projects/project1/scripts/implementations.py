import numpy as np

def least_squares(y,tx):
    """
    Linear regression using normal equations

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    """
    #Solve the linear system from normal equations
    w = np.linalg.solve(tx,y)

    #Compute loss
    loss = compute_loss_mse(y,tx,w)
    
    return(w,loss)

def ridge_regression(y, tx, lambda_):
    """
    Normal equations using L2 regularization

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param lambda_: float: penalty parameter
    """
    assert lambda_ > 0, "Step size gamma must be positive."
    
    #Compute Gram matrix
    gram = np.dot(tx.transpose(), tx)
    
    #Compute identity dxd matrix
    eye =  np.identity(tx.shape[1])

    #Compute lambda prime as lamda/2N
    plambda = lambda_/(2*tx.shape[0])

    #Solve the linear system from normal equation using L2 regularization
    w = np.dot( np.dot(np.linalg.inv(gram + plambda*eye), tx.transpose() ), y)

    #Compute loss
    loss = compute_loss_mse(y,tx,w)
    
    return(w,loss)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial modeo parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, initial_w)

    for n_iter in range(max_iters):
        # Compute gradient
        g = compute_gradient_mse(y, tx, w)

        # Update parameters according to gradient
        w = w  - gamma * g

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial modeo parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    n_iter = 0

    for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters):

        # Compute gradient for current batch
        gradient = compute_gradient_mse(batch_y, batch_tx, w)

        # Update model parameters
        w = w - gamma * gradient

        # Compute new loss
        loss = compute_loss_mse(y, tx, initial_w)

        print("Stochastic GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        n_iter += 1

        return w, loss


def compute_loss_mse(y, tx, w):
    """
    Calculate the MSE loss under weights vector w
    """
    n, d = tx.shape
    e = y - tx.dot(w)

    return e.dot(e)/(2*n)


def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    n, d = tx.shape

    e = y - tx.dot(w)

    return -tx.T.dot(e) / n


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
