Relevant code for project 1 is located here. The implemented functions are located in functions.py. The main.py file performs 
cross-validation for a given dataset, range of polynomial degrees and hyperparameters. A heatmap of the MSE as function of polynomial 
degree and hyperparameter is then shown, and the best parameter and (maximum) polynomial degree is also printed. The file takes three 
command line arguments: min_poly, max_poly and regression method. Cross-validation is performed from polynomial degree min_poly up to
degree max_poly - 1. Valid methods for regression are ols, ridge and lasso. Example:

python main.py 1 21 ridge

The file single_model.py is used for cases where we do not want to produce cross-validation, for example if we only want to perform a 
single prediction and produce a figure from the results.

The file special_case.py is reserved for performing a prediction on the Franke dataset, using OLS with maximum polynomial degree 5 in
x and y. The confidence intervals are then calculated and displayed in a figure.