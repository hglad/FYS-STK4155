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

def DataImport(filename, sc=10):
	# Load the terrain
	terrain1 = imread(filename)
	# Show the terrain
	downscaled = terrain1[0::sc,0::sc]

	# plt.figure()
	# plt.imshow(terrain1, cmap='gray')
	# plt.figure()
	# plt.imshow(downscaled, cmap='gray')
	return downscaled

def plot_surf(x,y,z, color, alpha=1):
	if len(x.shape) != 2:
		sqx = int(np.sqrt(len(x)))
		x = np.reshape(x, (sqx, sqx))
	if len(y.shape) != 2:
		sqy = int(np.sqrt(len(y)))
		y = np.reshape(y, (sqy, sqy))
	if len(z.shape) != 2:
		sqz = int(np.sqrt(len(z)))
		z = np.reshape(z, (sqz, sqz))

	# Framework for 3D plotting
	# fig = plt.figure()
	ax = plt.gca(projection='3d')

	# Customize the z-axis
	# ax.set_zlim(-0.1, 1.4)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	surf = ax.plot_surface(x, y, z, cmap=color, linewidth=0, antialiased=False, alpha=alpha)
	# ax.plot(x_p,y_p,z_p ,'-ko',alpha=1)
	# surf = ax.plot_surface(x_p, y_p, z_p, cmap=cm.ocean)
	# Add a color bar which maps values to colors
	# fig.colorbar(surf, shrink=0.5, aspect=5)

def plot_points(x,y,z):
	if len(x.shape) != 2:
		sqx = int(np.sqrt(len(x)))
		x = np.reshape(x, (sqx, sqx))
	if len(y.shape) != 2:
		sqy = int(np.sqrt(len(y)))
		y = np.reshape(y, (sqy, sqy))
	if len(z.shape) != 2:
		sqz = int(np.sqrt(len(z)))
		z = np.reshape(z, (sqz, sqz))
	ax = plt.gca(projection='3d')
	ax.plot_wireframe(x, y, z)
	# ax.plot(np.ravel(x),np.ravel(y),np.ravel(z), '-ko', alpha=1)


def cross_validation(x, y, z, k, p, param=0.1, method='ols'):
	kfold = KFold(n_splits = k, shuffle=True)
	len_beta = int((p+1)*(p+2)/2)

	beta_array = np.zeros( (k, len_beta) )
	error_test = bias_test = var_test = 0
	error_train = bias_train = var_train = 0
	R2_sum = MSE_test = MSE_train = var_beta = 0
	beta_var = np.zeros(len_beta)

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
		"""!!!!"""
		# z_test_1d = np.ravel(FrankeFunction(x_test_k, y_test_k))
		"""!!!!"""
		# z_train_k = np.reshape(z_train_k, (len(y_train_k), len(x_train_k)))
		# print (z_train_k.shape)
		# plt.imshow(z_train_k, cmap='gray')
		# plt.show()

		if (method == 'ols'):
			# Compute model with train data from fold
			X_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)
			beta_k = beta_ols(X_k, z_train_1d)

			# Predict with trained model using test data from fold
			X_test_k = CreateDesignMatrix_X(x_test_k, y_test_k, p)

			# Predict with trained model on train data
			X_train_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)

			z_pred_test = X_test_k @ beta_k
			z_pred_train = X_train_k @ beta_k

		if (method == 'ridge'):
			X_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)
			X_mean = np.mean(X_k, axis=0)
			z_train_mean = np.mean(z_train_k)

			X_k = X_k - X_mean
			z_test_k -= z_train_mean
			z_test_1d-= z_train_mean

			beta_k = beta_ridge(X_k, z_train_1d, param)

			# Predict with trained model using test data from fold
			X_test_k = CreateDesignMatrix_X(x_test_k, y_test_k, p) - X_mean
			z_pred_test = X_test_k @ beta_k + z_train_mean

			# Predict with trained model on train data
			X_train_k = CreateDesignMatrix_X(x_train_k, y_train_k, p) - X_mean
			z_pred_train = X_train_k @ beta_k + z_train_mean

		if (method == 'lasso'):
			model = linear_model.Lasso(alpha=param, fit_intercept=False, max_iter=5000)
			X_test_k = CreateDesignMatrix_X(x_test_k, y_test_k, p)
			X_train_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)

			z_ = np.ravel(z_train_k)
			lasso = model.fit(X_train_k, z_)
			z_pred_test = lasso.predict(X_test_k)
			z_pred_train = lasso.predict(X_train_k)

		# Compute error, bias, variance
		error_test += np.mean((z_test_1d - z_pred_test)**2)
		bias_test += np.mean( (z_test_1d - np.mean(z_pred_test))**2 )
		var_test += np.var(z_pred_test)

		error_train += np.mean((z_train_1d - z_pred_train)**2)
		# bias_train += np.mean( (z_train_1d - np.mean(z_pred_train))**2 )
		# var_train += np.var(z_pred_train)

		# Compute R2 and MSE scoores
		R2_sum += metrics.r2_score(z_test_1d, z_pred_test)
		MSE_test += metrics.mean_squared_error(z_test_1d, z_pred_test)
		MSE_train += metrics.mean_squared_error(z_train_1d, z_pred_train)

		i += 1

	# Return mean of all MSEs for all folds
	# var_test = np.mean(np.var(z_preds_test_array,axis=1, keepdims=True))
	# var_train = np.mean(np.var(z_preds_train_array, axis=1, keepdims=True))

	return R2_sum/k, MSE_test/k, MSE_train/k, error_test/k, \
	bias_test/k, var_test/k, error_train/k

# Create design matrix, find beta and predict

def predict_lasso(x, y, z, p, alpha, max_iter=5000):
	model = linear_model.Lasso(alpha=alpha, fit_intercept=True, max_iter=max_iter)
	X = CreateDesignMatrix_X(x, y, p)
	z_ = np.ravel(z)
	lasso = model.fit(X, z_)
	z_pred = lasso.predict(X)
	return z_, z_pred

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
		model = linear_model.Lasso(alpha=param, fit_intercept=False, max_iter=5000)
		lasso = model.fit(X, z_)
		z_pred = lasso.predict(X)

	return z_, z_pred

def beta_ols(X, z):
	beta = np.linalg.pinv( np.dot(X.T, X)) .dot(X.T) .dot(z)
	return beta

def beta_ridge(X, z, l):
	m = len(X[0,:])
	lmbd = l*np.eye(m)
	beta = np.linalg.pinv( np.dot(X.T, X) + lmbd) .dot(X.T) .dot(z)
	return beta

def Franke_dataset(n, noise=0.5):
	x = np.linspace(0, 1, n)
	y = np.linspace(0, 1, n)
	x, y = np.meshgrid(x,y)

	eps = np.asarray([np.random.normal(0,noise,n*n)])
	eps = np.reshape(eps, (n,n))
	z = FrankeFunction(x,y) + eps

	return x, y, z

def CI(x, sigma, t=1.96):
	CI_low = x - t*sigma
	CI_high = x + t*sigma

	plt.plot(range(len(x)), x)
	plt.plot(range(len(x)), CI_low)
	plt.plot(range(len(x)), CI_high)
	plt.show()
	# return CI_low, CI_high


def plot_bias_var_err(polys, bias_test, var_test, MSE_test, MSE_train):
	plt.plot(polys, bias_test, '--b', label='bias (test)')
	plt.plot(polys, var_test,  '--r', label='variance (test)')

	plt.plot(polys, MSE_test,  '--g', label='MSE (test)')
	plt.plot(polys, MSE_train,  '-g', label='MSE (train)')

	plt.legend()
	plt.show()

def plot_mse_train_test(polys, MSE_test, MSE_train):
	plt.plot(polys, MSE_test, '--r', label='MSE (test)')
	plt.plot(polys, MSE_train, '-b', label='MSE (train)')
	plt.legend()
	plt.show()


#
