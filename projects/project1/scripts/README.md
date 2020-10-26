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

Describe models in `implementations.py` and `implementation_variants.py`
Needs to include: high-level description of the loss function, the optimisation method, additional parameters (for instance, terminating condition for GD methods)

The following models are implemented in `implementations.py`:
- `least_squares` : computes the weight vector by solving the normal equations. It returns the weight vector and the loss, computed as Mean Squared Error. The Mean Squared Error computes the squared power of the difference between the value of 'y' and the product between the input data 'x' and the weight vector 'w'.
- `ridge_regression` : computes the weight vector by solving the normal equation under L2 regularization. That is, penalizing the square of the norm of the vector 'w'. The function accepts an hyperparameter 'lambda_' that is the penalty factor. It can be tuned differently to set the severity of the penalization.
- `least_squares_GD` : solves the least squares problem with the Gradient Descent algorithm (an iterative optimization algotithm based on first-order approximation of the loss function(MSE)). The function takes as additional parameters :'max_iter'(maximum number of iteration), 'gamma' (step size of Gradient Descent), 'threshold' (it sets the minimum difference for the loss computed by two consecutive iterations of the algorithm. If the difference between the two values is below this parameter, the algorithm ends), 'verbose' (bool value to output the steps of the algorithm. Set to false by default). If the loss computed at one step is equal to Nan, the algorithm ends.
- `least_squares_SGD` : Stochastic Gradient Descent is a variant of the GD algorithm where at each step the gradient value and the weight vector 'w' are updated considering a subgroup of the dataset(batch). By default, the batch size is set to 1, meaning that at each step the algorithm picks at random one data sample.
- `logistic_regression` : applies the GD algorithm trying to minimize the loss function, computed as negative Log Likelihood (NNL).
It takes the same parameters as 'least_squares_GD'. Additionally it checks whether the provided input 'y' is a binary set of labels 
{1;0}.
### Experiment setup

Describe how functions in `utils.py` and `data_utils.py` can be used to test different models on features (cross-val and splitting)  





