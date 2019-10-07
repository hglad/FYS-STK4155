from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn.metrics as metrics
import sklearn.model_selection as mselect
from sklearn.model_selection import KFold
from sklearn import linear_model
from imageio import imread
import sys
import seaborn as sb
import pandas as pd


def CreateDesignMatrix_X(x, y, p = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
    # FLatten multidimensional array into one array
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	len_beta = int((p+1)*(p+2)/2)		# Number of elements in beta
	X = np.ones((N,len_beta))

	for i in range(1,p+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.  - 0.1*(9*y+1))
    term3 = 0.50*np.exp(-(9*x-7)**2 / 4.   - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2        - (9*y-7)**2)

    return term1 + term2 + term3 + term4


def Franke_dataset(n, noise=0.5):
	x = np.linspace(0, 1, n)
	y = np.linspace(0, 1, n)
	x, y = np.meshgrid(x,y)

	eps = np.asarray([np.random.normal(0,noise,n*n)])
	eps = np.reshape(eps, (n,n))
	z = FrankeFunction(x,y) + eps

	return x, y, z


def DataImport(filename, sc=10):
	# Load the terrain data
	data = imread(filename)

	# Scale the terrain data
	downscaled = data[0::sc,0::sc]
	print ("Dataset scaled from", data.shape, "to", downscaled.shape)

	return downscaled


def cross_validation(x, y, z, k, p, dataset, param=0.1, method='ols', penalize_intercept=False):
	"""
	Perform K-fold cross-validation given dataset, method for regression and
	hyperparameter in the case of ridge or lasso.
	"""
	kfold = KFold(n_splits = k, shuffle=True)
	len_beta = int((p+1)*(p+2)/2)
	# Initialiation of variables
	error_test = bias_test = var_test = 0
	error_train = bias_train = var_train = 0
	R2_sum = MSE_test = MSE_train = var_beta = 0

	i = 0
	for train_inds, test_inds in kfold.split(x):
		x_train_k = x[train_inds]
		y_train_k = y[train_inds]
		z_train_k = z[train_inds]
		z_train_1d = np.ravel(z_train_k)

		x_test_k = x[test_inds]
		y_test_k = y[test_inds]
		z_test_k = z[test_inds]
		z_test_1d = np.ravel(z_test_k)

		if (dataset == 'Franke'):
			z_true_test_1d = np.ravel(FrankeFunction(x_test_k, y_test_k))
			z_true_train_1d = np.ravel(FrankeFunction(x_train_k, y_train_k))

		X_test_k = CreateDesignMatrix_X(x_test_k, y_test_k, p)
		X_train_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)

		if (method == 'ols'):
			beta_k = beta_ols(X_train_k, z_train_1d)
			z_pred_test = X_test_k @ beta_k
			z_pred_train = X_train_k @ beta_k

		if (method == 'ridge'):
			# Penalize_intercept attempts to remove the intercept. Not used.
			if (penalize_intercept == True):
				X_mean = np.mean(X_train_k, axis=0)
				z_train_mean = np.mean(z_train_k)

				X_train_k = X_train_k - X_mean
				z_test_k -= z_train_mean
				z_test_1d-= z_train_mean
				z_train_1d -= z_train_mean
				z_train_k -= z_train_mean

				# Predict with trained model using test data from fold
				beta_k = beta_ridge(X_train_k, z_train_1d, param, len_beta)
				X_test_k = X_test_k - X_mean
				z_pred_test = X_test_k @ beta_k + z_train_mean

				# Predict with trained model on train data
				X_train_k = X_train_k - X_mean
				z_pred_train = X_train_k @ beta_k + z_train_mean
			else:
				beta_k = beta_ridge(X_train_k, z_train_1d, param, len_beta)
				z_pred_test = X_test_k @ beta_k
				z_pred_train = X_train_k @ beta_k

		if (method == 'lasso'):
			# Generate linear model using Lasso and perform predictions
			model = linear_model.Lasso(alpha=param, fit_intercept=False, tol=0.01, max_iter=30000)
			lasso = model.fit(X_train_k, z_train_1d)
			z_pred_test = lasso.predict(X_test_k)
			z_pred_train = lasso.predict(X_train_k)

		# Compute error, bias, variance
		error_test += np.mean((z_test_1d - z_pred_test)**2)
		bias_test += np.mean( (z_test_1d - np.mean(z_pred_test))**2 )
		var_test += np.var(z_pred_test)
		error_train += np.mean((z_train_1d - z_pred_train)**2)

		# Compute R2 and MSE scoores
		if (dataset == 'Franke'):
			R2_sum += metrics.r2_score(z_true_test_1d, z_pred_test)
			MSE_test += metrics.mean_squared_error(z_true_test_1d, z_pred_test)
			MSE_train += metrics.mean_squared_error(z_true_train_1d, z_pred_train)
		else:
			R2_sum += metrics.r2_score(z_test_1d, z_pred_test)
			MSE_test += metrics.mean_squared_error(z_test_1d, z_pred_test)
			MSE_train += metrics.mean_squared_error(z_train_1d, z_pred_train)

		i += 1

	return R2_sum/k, MSE_test/k, MSE_train/k, error_test/k, \
	bias_test/k, var_test/k, error_train/k


# Create design matrix, find beta and predict
def predict_poly(x, y, z, p, param=0, method='ols'):
	X = CreateDesignMatrix_X(x, y, p)
	z_ = np.ravel(z)

	if (method == 'ols'):
		beta = np.linalg.pinv(np.dot(X.T, X)) .dot(X.T) .dot(z_)
		z_pred = X @ beta

	if (method == 'ridge'):
		m = len(X[0,:])
		lmbd = param*np.eye(m)
		beta = np.linalg.pinv(np.dot(X.T, X) + lmbd) .dot(X.T) .dot(z_)
		z_pred = X @ beta

	if (method == 'lasso'):
		model = linear_model.Lasso(alpha=param, fit_intercept=False, tol=0.01, max_iter=30000)
		lasso = model.fit(X, z_)
		z_pred = lasso.predict(X)

	return z_, z_pred


def beta_ols(X, z):
	beta = np.linalg.pinv( np.dot(X.T, X)) .dot(X.T) .dot(z)
	return beta


def beta_ridge(X, z, param, m):
	lmbd = param*np.eye(m)
	beta = np.linalg.pinv( np.dot(X.T, X) + lmbd) .dot(X.T) .dot(z)
	return beta


def CI(x, sigma, nx, ny, p, t=1.96):
	"""
	Compute 95% confidence interval (CI) for coefficients beta. Plot the
	estimated beta values and their corresponding CIs.
	"""
	CI_low = x - t*sigma
	CI_high = x + t*sigma
	plot_range = range(len(x))
	CI_ = np.zeros((len(x), 2))

	for i in range(len(x)):
		CI_[i,0] = CI_low[i]
		CI_[i,1] = CI_high[i]

	for i in range(len(x)):
		plt.plot([plot_range[i], plot_range[i]], CI_[i], '-ko', alpha=0.8, markersize=5)
		plt.plot(plot_range[i], x[i], 'k_', markersize=10)

	plt.ylabel('Confidence Intervals')
	plt.xlabel('index', rotation=0)
	plt.show()


def UnitTest():
	"""
	Unit test to make sure that regular OLS gives the same results as ridge
	regression with hyperparameter set to 0
	"""
	p = 5
	len_beta = int((p+1)*(p+2)/2) # Number of elements in beta (!= p generally)

	x,y,z = Franke_dataset(50,0)
	X = CreateDesignMatrix_X(x,y,p)
	z_1d = np.ravel(z)

	B_OLS = beta_ols(X, z_1d)
	B_Ridge = beta_ridge(X, z_1d, 0, len_beta)
	eps = 1e-15
	for i in range(len(B_OLS)):
		assert B_OLS[i]-B_Ridge[i] < eps


def plot_surf(x,y,z, color, alpha=1):
	# Framework for 3D plotting
	fig = plt.figure()
	ax = plt.gca(projection='3d')

	# Customize the z-axis
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('$x$', fontsize=20)
	ax.set_ylabel('$y$', fontsize=20)
	ax.set_zlabel('$z$', fontsize=20)
	surf = ax.plot_surface(x, y, z, cmap=color, linewidth=0, antialiased=False, alpha=alpha, shade=True)

	# Add a color bar which maps values to colors
	fig.colorbar(surf, shrink=0.5, aspect=5)

def plot_bias_var_err(polys, bias_test, var_test, MSE_test, MSE_train):
	# Plot bias, variance and MSE as function of model complexity.
	plt.plot(polys[0],MSE_test[0,0], '-b')		# dummy plots for legend
	plt.plot(polys[0],MSE_test[0,0], '-r')
	plt.plot(polys[0],MSE_test[0,0], '-g')
	plt.legend()
	plt.xlabel('Complexity')

	plt.plot(polys, bias_test, '-b', label='bias (test)')
	plt.plot(polys, var_test,  '-r', label='variance (test)')
	plt.plot(polys, MSE_test,  '-g', label='MSE (test)')

	plt.show()

def plot_mse_train_test(polys, MSE_test, MSE_train, params, nx, ny):
	# Plot train and test MSE as function of model complexity.
	plt.plot(polys[0],MSE_test[0,0], '-r')		# dummy plots for legend
	plt.plot(polys[0],MSE_test[0,0], '-b')
	plt.tight_layout()
	plt.legend(['MSE (test data)', 'MSE (train data)'])

	plt.xlabel('Complexity')
	plt.ylabel('MSE')

	plt.plot(polys, MSE_test, '-r', label='MSE (test)',alpha=1)
	plt.plot(polys, MSE_train, '-b', label='MSE (train)',alpha=1)

	plt.show()

def plot_mse_poly_param(params, polys, MSE):
	x, y = np.meshgrid(params,polys)
	ax = plt.gca(projection='3d', figsize=(8,8))
	fig = plt.figure(figsize=(6,6))
	# ax.plot_surface(x, y, MSE, cmap=cm.coolwarm)
	ax.set_xlabel('Parameter')
	ax.set_ylabel('Complexity')
	ax.set_zlabel('MSE')
	plt.show()

def plot_heatmap(params, polys, MSE):
	# plt.rcParams['figure.figsize'] = 8,8
	plt.rcParams.update({'font.size': 16})
	plt.imshow(MSE.T, cmap=cm.viridis)
	plt.tight_layout()

	ylabels = ['{:,.1e}'.format(i) for i in params]
	plt.yticks(np.linspace(0, len(params)-1, len(params)), ylabels)
	plt.xticks(np.arange(0, abs(polys[0]-polys[-1])+1, 1), polys)

	cbar = plt.colorbar()
	cbar.set_label('MSE')

	plt.xlabel('Max. polynomial degree')
	plt.ylabel('Hyperparameter')

	plt.show()


def terrain_2d(x,y,z):
	plt.figure()
	plt.imshow(z, cmap=cm.terrain)
	plt.colorbar()
	plt.clim(0,1)

	plt.show()


#
