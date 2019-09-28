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

	return downscaled

def plot_surf(x,y,z, color, alpha=1):

	# Framework for 3D plotting
	fig = plt.figure()
	ax = plt.gca(projection='3d')

	# Customize the z-axis
	# ax.set_zlim(-0.1, 1.4)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('$x$', fontsize=20)
	ax.set_ylabel('$y$', fontsize=20)
	ax.set_zlabel('$z$', fontsize=20)
	surf = ax.plot_surface(x, y, z, cmap=color, linewidth=0, antialiased=False, alpha=alpha, shade=True)

	# Add a color bar which maps values to colors
	fig.colorbar(surf, shrink=0.5, aspect=5)

def plot_pred(x,y,z):

	# if len(x.shape) != 2:
	# 	sqx = int(np.sqrt(len(x)))
	# 	x = np.reshape(x, (sqx, sqx))
	# if len(y.shape) != 2:
	# 	sqy = int(np.sqrt(len(y)))
	# 	y = np.reshape(y, (sqy, sqy))
	# if len(z.shape) != 2:
	# 	sqz = int(np.sqrt(len(z)))
	# 	z = np.reshape(z, (sqz, sqz))

	ax = plt.gca(projection='3d')
	ax.plot_wireframe(x, y, z, ccount=10, rcount=10)
	# ax.plot(np.ravel(x),np.ravel(y),np.ravel(z),'-ko')


def cross_validation(x, y, z, k, p, dataset, param=0.1, method='ols', penalize_intercept=False):
	kfold = KFold(n_splits = k, shuffle=True)
	len_beta = int((p+1)*(p+2)/2)

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
			beta_k = np.linalg.pinv( np.dot(X_train_k.T, X_train_k)) .dot(X_train_k.T) .dot(z_train_1d)
			z_pred_test = X_test_k @ beta_k
			z_pred_train = X_train_k @ beta_k

		if (method == 'ridge'):
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
			model = linear_model.Lasso(alpha=param, warm_start=True, fit_intercept=False, tol=0.01, max_iter=20000)
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

def beta_ridge(X, z, param, m):
	lmbd = param*np.eye(m)
	beta = np.linalg.pinv( np.dot(X.T, X) + lmbd) .dot(X.T) .dot(z)
	return beta

def CI(x, sigma, nx, ny, p, t=1.96):
	CI_low = x - t*sigma
	CI_high = x + t*sigma
	plot_range = range(len(x))
	CI_ = np.zeros((len(x), 2))

	for i in range(len(x)):
		CI_[i,0] = CI_low[i]
		CI_[i,1] = CI_high[i]

	for i in range(len(x)):
		plt.plot([plot_range[i], plot_range[i]], CI_[i], '-ko', alpha=0.8, markersize=5)
		plt.plot(plot_range[i], x[i], 'k|', markersize=10)

	# plt.title('Confidence Interval for $\\beta$\n%d x %d grid, p = %d' % (nx,ny,p))
	plt.ylabel('Confidence Intervals')
	plt.xlabel('index', rotation=0)
	plt.show()

def plot_bias_var_err(polys, bias_test, var_test, MSE_test, MSE_train):
	plt.plot(polys, bias_test, '--b', label='bias (test)')
	plt.plot(polys, var_test,  '--r', label='variance (test)')

	plt.plot(polys, MSE_test,  '--g', label='MSE (test)')
	plt.plot(polys, MSE_train,  '-g', label='MSE (train)')

	# plt.legend()
	plt.show()

def plot_mse_train_test(polys, MSE_test, MSE_train, params, nx, ny):
	plt.plot(polys[0],MSE_test[0,0], '-r')		# dummy plots for legend
	plt.plot(polys[0],MSE_test[0,0], '-b')

	plt.legend(['MSE (test data)', 'MSE (train data)'])
	# plt.title('MSE, Lasso Regression\n%d x %d grid' % (nx,ny))
	plt.xlabel('Complexity')
	plt.ylabel('MSE')

	plt.plot(polys, MSE_test, '-r', label='MSE (test)',alpha=0.5)	# 0.5
	plt.plot(polys, MSE_train, '-b', label='MSE (train)',alpha=0.3) # 0.3

	# Label the different plots with their lambda value
	# plt.axis([0, polys[-1]+2, 0.23, 0.35])
	# for i in range(len(params)):
	# 	plt.text(polys[-1], MSE_test[-1][i], "$\\lambda$ = %1.1e" % params[i])

	plt.show()

def plot_mse_poly_param(params, polys, MSE):
	x, y = np.meshgrid(params,polys)
	ax = plt.gca(projection='3d')

	ax.plot_surface(x, y, MSE, cmap=cm.coolwarm)
	ax.set_xlabel('Parameter')
	ax.set_ylabel('Complexity')
	ax.set_zlabel('MSE')
	plt.show()

#
