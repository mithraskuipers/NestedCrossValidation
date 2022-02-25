# Nested Cross Validation

Manual Python implementation of the Nested Cross Validation algorithm (`nested_cv.py`), performing exhaustive search over the specified parameter values. It estimates the generalization error (`MSE`) of the underlying model and its (hyper)parameter search. Generally, choosing the parameters that maximize non-nested Cross Validation biases the model to the dataset, yielding an overly-optimistic score. Due to the nature of this exhaustive search, the time complexity is O(n * m).

~ Mithras

![](https://raw.githubusercontent.com/mithraskuipers/NestedCrossValidation/main/img/nCV.png)

## Return values

| Name | Type | Description |
| :- |:-|:-|
`ncv_error` | `float64` | Unbiased estimate of the prediction error when using the best hyperparameters (after exhaustive search). It is the average of 'outer_cv_scores'. |
`outer_cv_scores` | `list` of _k_ `float64` | Estimates of the prediction error for each fold of the outer CV. |

## Parameters

| Parameter | Type/Values | Description |
| :- |:-|:-|
| `X` | `pandas DataFrame` | input vector(s) without missing values. |
| `y` | `pandas Series` | output vector without missing values. |
| `estimator` |  scikit-learn estimator object | The base object of sci-kit learn which implements a fit method to learn from data. |
| `param_dict` | `dict` | Dictionary with parameter names (str) as keys and lists of parameter settings to try as values. The script performs exhaustive search to identify the best combination of hyperparameters. |
| `ncv_n_outerfolds` | `int`, default = 3 | Number of folds for the outer Cross Validation. |
| `ncv_n_innerfolds` | `int`, default = 3 | Number of folds for the inner Cross Validation. |
| `verbose` | `bool` or `int`, default = `False` | If `True`, it prints the best hyperparameter(s) for each outer fold and the unbiased estimate of the prediction error when using the best hyperparameters (i.e. the return value). |

## Dependencies

| Import | Install |
| :- |:-|
| import operator | N/A (built-in) |
| import pandas | pip install pandas |
| import numpy as np | pip install numpy |
| from sklearn import model_selection | pip install sklearn |
| from sklearn.metrics import mean_squared_error | pip install sklearn |
| from sklearn.model_selection import ParameterGrid | pip install sklearn |

## Synopsis

`nested_cv(X, y, estimator, param_dict, ncv_n_outerfolds, ncv_n_innerfolds, verbose = True)`

## Example

```
from sklearn.datasets import load_iris
from sklearn.svm import SVC
iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.Series(iris.target)
estimator = SVC(kernel="rbf")
param_dict = {"C": [1, 10, 100],
              "gamma": [.01, .1]}
ncv_n_outerfolds = 3
ncv_n_innerfolds = 2

ncv_error, outer_cv_scores = nested_cv(X, y, estimator, param_dict, ncv_n_outerfolds, ncv_n_innerfolds, verbose = True)
```

### Output

```
Best hyperparams of Outer fold 0:
{'C': 1, 'gamma': 0.01}

Best hyperparams of Outer fold 1:
{'C': 1, 'gamma': 0.1}

Best hyperparams of Outer fold 2:
{'C': 1, 'gamma': 0.01}

Unbiased prediction error (MSE) when using SVC is:
0.48
```

## TODOs

1. Support multiprocessing (i.e. leveraging multiple processors)
2. Option for easily changing scoring function(s) (separately for outer and inner CV loops)
3. Computing multiple error measures and return in DataFrame format.
4. Support NumPy arrays instead of Pandas DataFrame/Series.
5. Plotting functionality?
6. Pipeline to next step: Actual model creation for predictions.
