# Learning to Discover - The Higgs boson challenge
## CS-433 Machine Learning Project 1 - Fall 2020

This repository contains all code used in the Project 1 of the Fall 2020 Machine Learning course (CS-433)

### Structure

The code is structured in the following way:

- `implementations.py` contains implementations of the main functions required for Project 1 (see [project description](https://raw.githubusercontent.com/epfml/ML_course/master/projects/project1/project1_description.pdf)).
- `implementation_variants.py` contains variants of the main models such as, for instance, a lasso regression model under subgradient descent, 
- `costs.py` contains implementations of various common cost functions. This file contains dependencies for `implementations.py`.
- `data_utils.py` contains helper functions for transforming and pre-processing data. This file contains dependencies for `implementations.py`.
- `utils.py` contains helper functions for model training and selection, such as data splitting and test data generation.
- `plots.py` contains functions to visualise the output of model training or evaluation. 

### Example run
To re-generate the submission in `final_submisison.csv` run

```python run.py``` 




