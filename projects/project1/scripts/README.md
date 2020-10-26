# Learning to Discover - The Higgs boson challenge
## CS-433 Machine Learning Project 1 - Fall 2020

This repository contains all code used in the Project 1 of the Fall 2020 Machine Learning course (CS-433)

### Structure

The code is structured in the following way:

- `implementations.py` contains implementations of the main functions required for Project 1 (see [project description](https://raw.githubusercontent.com/epfml/ML_course/master/projects/project1/project1_description.pdf)).
- `implementation_variants.py` contains variants of the main models such as, for instance, a lasso regression model under subgradient descent.
- `costs.py` contains implementations of various common cost functions. This file contains dependencies for `implementations.py`.
- `utils.py` contains helper functions for model training and selection, such as data splitting and test data generation. This file contains dependencies for `implementations.py`.
- `data_utils.py` contains helper functions for transforming and pre-processing data.
- `plots.py` contains functions to visualise the output of model training or evaluation. 

### Re-producing experiments
To re-generate the submission in `final_submisison.csv` run

```python run.py``` 

The script expects two files `train.csv` and `test.csv` to be in a folder called `data` that is in the same parent directory as `run.py`.

The notebook `Run.ipynb` contains all experiments described in the final project report. In addition, it contains results for some models not described in detail in the report.
The experiments in `Run.ipynb` include selection of the best performing feature set and optimisation of model training hyperparameters such as ideal step size `GAMMA` for logistic regression gradient descent.

The notebook `Data Exploration.ipynb` contains a quick exploration of the data.
It was used to select the least informative features with the lowest linear correlation with the target variable hard-coded as `LEAST_INFO` in `data_utils.py`.

### Models

The following linear classification methods are implemented in `implementations.py`:
- `least_squares` : Computes the weight vector `w` by solving the normal equations. The returned model weights minimised the mean sqaured error loss between the target labels `y` and the dot product between the input data `x` and the weight vector `w`.
- `ridge_regression` : Computes the weight vector `w` by solving the normal equation under L2 regularization. That is, penalizing the square of the norm of the vector `w`.
The function accepts an hyperparameter `lambda_` that is the penalty factor. `lambda_` can be used to guide the trade-off between parameter shrinkage and the optimal MSE solution.
- `least_squares_GD` : Solves the least-squares problem using gradient descent (GD) (an iterative optimization algotithm based on first-order approximation of the loss function).
The step size of each update step is set by the input parameter `gamma`.
The optimisation procedure is either terminated after the maximum number of iterations is reached (`max_iter`) or due to loss convergence (loss does not change by more than `threshold` between two consecutive steps).
- `least_squares_SGD` : Stochastic Gradient Descent is a variant of the GD algorithm where at each step the gradient value and the weight vector `w` are updated considering a subgroup of data records (batch).
By default, the batch size is set to 1, meaning that at each step the algorithm picks at random one data sample.
- `logistic_regression` : Runs gradients descent to minimize the negative log-likelihood (NLL) loss of a logistic regression model.
It takes the same parameters as `least_squares_GD`. Additionally it checks whether the provided input `y` is a binary set of labels  in `{0, 1}`.
- `logistic_regression`: Applies the GD algorithm to minimize the NLL under L2 regularization. It takes the same parameters as `logistic_regression`, plus the penalization parameter `lambda_`.

`implementation_variants.py` contains some variants of the standard implementations of the models described above:
- `lasso_GD` : Implements (sub-)GD for Lasso regression. The model finds the minimum MSE weights under the constraint of  minimising the 1-norm  of the weights vector `w`.
- `logistic_regression_SGD` : SGD for finding the minimum NLL weights under a `logistic_regression` model. See `least_squares_SGD` for details on input parameters.
- `logistic_regression_mean` : Variant of `logistic_regression` in which loss and gradient value are normalized to the size of the training set. See `logistic_regression` for detail.
- `least_squares_SGD_robbinson` : Variant of `least_squares_SGD` using a non-constant step size for GD which decreases at each step.
The hyperparameter defining the stepsize is `r_gamma`, and it must be in range `[0.5,1]`. See `least_squares_SGD` for details.

### Helper functions for data processing and experiment setup  
In `data_utils.py` many useful functions for feature standardization and transformation are defined. The main functions are:
-`feature_transform_mostinfo` : Takes the input data `x` as input. Removes the 10 least informative features from the input data.
-`feature_transform_imputejet` : Takes the input data `x` as input. Imputes undefined values in the input data by setting such values to the mean of defined values in the data sample.
-`feature_transform_polybasis` : Takes the input data `x` as input. Performs feature expansion to a polynomial basis of maximum degree `degree`. It automatically handles the presence of undefined values among features.

In `utils.py` functions for loading data from `.csv` files are provided.
Moreover functions to compute prediction accuracy (see `eval_model`), generate prediction on  a test dataset (`predict_labels`), and create `.csv` submission are defined. For details on parameters, and label encodings see the docstring of `eval_model` and `predict_labels`.

There are also functions which are useful when training and validating models, such as:
- `train_eval_split` : Takes as inputs dataset `y` and `tx`, in addition to `split_ratio` and `seed`. It randomly splits the data set in two sets, train and evaluation, according to the ratio defined by `split_ratio`.
- `k_fold_iter` : returns a k-fold iterator over a dataset `(y, tx)` in the form of `(train_split, test_split)`. It takes as additional parameters `k_fold`, the number of iteration of the validation algorithm, and `seed` for random shuffle of data samples. 





