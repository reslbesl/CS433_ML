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
The following models are implemented in `implementations.py`:
- `least_squares` : computes the weight vector by solving the normal equations. It returns the weight vector and the loss, computed as Mean Squared Error. The Mean Squared Error computes the squared power of the difference between the value of 'y' and the product between the input data 'x' and the weight vector 'w'.
- `ridge_regression` : computes the weight vector by solving the normal equation under L2 regularization. That is, penalizing the square of the norm of the vector 'w'. The function accepts an hyperparameter 'lambda_' that is the penalty factor. It can be tuned differently to set the severity of the penalization.
- `least_squares_GD` : solves the least squares problem with the Gradient Descent algorithm (an iterative optimization algotithm based on first-order approximation of the loss function(MSE)). The function takes as additional parameters :'max_iter'(maximum number of iteration), 'gamma' (step size of Gradient Descent), 'threshold' (it sets the minimum difference for the loss computed by two consecutive iterations of the algorithm. If the difference between the two values is below this parameter, the algorithm ends), 'verbose' (bool value to output the steps of the algorithm. Set to false by default). If the loss computed at one step is equal to Nan, the algorithm ends.
- `least_squares_SGD` : Stochastic Gradient Descent is a variant of the GD algorithm where at each step the gradient value and the weight vector 'w' are updated considering a subgroup of the dataset(batch). By default, the batch size is set to 1, meaning that at each step the algorithm picks at random one data sample.
- `logistic_regression` : applies the GD algorithm trying to minimize the loss function, computed as negative Log Likelihood (NNL).
It takes the same parameters as 'least_squares_GD'. Additionally it checks whether the provided input 'y' is a binary set of labels 
{1;0}.
- `logistic_regression`: applies the GD algorithm to minimize the NNL under L2 regularization. It takes the same parameters as 'logistic_regression', plus the penalization parameter 'lambda_'.

Furthermore there are some variants to the standard implementations of the mentioned models. They are defined in `implementation_variants.py`:
- `lasso_GD` : implements GD under Lasso regularization. That is, penalizing the norm 1 of the vector 'w'. See 'ridge_regression' for details.
- `logistic_regression_SGD` : Applies SGD to the 'logistic_regression' model. Contrary to the 'least_squares_SGD', the loss function is NNL. See 'least_squares_SGD' for details.
- `logistic_regression_mean` : variant of 'logistic_regression' where loss and gradient value are normalized to the size of the training set. See 'logistic_regression' for detail
- `least_squares_SGD_robbinson` : variant of 'least_squares_SGD' using a non-costant step size for Gradiaent Descent which decreases at each step. The hyperparameter defining the stepsize is 'r_gamma', and it must be in range [0.5,1]. See 'least_squares_SGD' for details.

### Experiment setup 
In `data_utils.py` many useful functions for feature standardization and transformation are defined:
-`feature_transform_mostinfo` : Takes the input data 'x' as input. Removes the 10 least informative features from the input data.
-`feature_transform_imputejet` : Takes the input data 'x' as input. Imputes undefined values in the input data by setting such values to the mean of all the features in a data sample.
-`feature_transform_polybasis` : Takes the input data 'x' as input. Performs feature expantion on polynomial basis. It handles the presence of undefined values among features. It takes also an int parameter 'degree' that is the degree of the polynomial.

In `utils.py` functions for loading data from '.csv' files are provided. Moreover functions to check the accuracy of the model, generate prediction on test dataset, and create '.csv' submission are defined. There are also functions which are useful when training and validating models, such as:
- `train_eval_split` : Takes as inputs dataset 'y' and 'tx', in addition to 'split_ratio' and 'seed'. It randomly splits the data set in two sets, train and evaluation, according to parameter 'split_ratio'. The split is automatically performed in `load_data`.
- `k_fold_iter` : returns a k-fold iterator for the dataset (y, tx) in the form of (train_split, test_split) . It is used in k-fold cross validation. It takes as additional parameters 'k_fold', the number of iteration of the validation algorithm, and 'seed' for random shuffle of data samples. 





