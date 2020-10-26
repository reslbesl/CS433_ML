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
- `lasso_GD` : Implements (sub-)GD for Lasso regression. The model finds the minimum MSE weights under the constraints of  minimising the 1-norm  of the weights vector `w`.
- `logistic_regression_SGD` : Applies SGD to the 'logistic_regression' model. Contrary to the 'least_squares_SGD', the loss function is NNL. See 'least_squares_SGD' for details.
- `logistic_regression_mean` : variant of 'logistic_regression' where loss and gradient value are normalized to the size of the training set. See 'logistic_regression' for detail
- `least_squares_SGD_robbinson` : variant of 'least_squares_SGD' using a non-costant step size for Gradiant Descent which decrease at each step. The hyperparameter defining the stepsize is 'r_gamma', and it must be in range [0.5,1]. See 'least_squares_SGD' for details.

### Experiment setup

Describe how functions in `utils.py` and `data_utils.py` can be used to test different models on features (cross-val and splitting)  





