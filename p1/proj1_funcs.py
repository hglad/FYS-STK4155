from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn.metrics as metrics
import sklearn.model_selection as mselect
from sklearn.model_selection import KFold

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

def plot_surf(x,y,z):
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
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Customize the z-axis
	# ax.set_zlim(-0.1, 1.4)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Add a color bar which maps values to colors
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

def cross_validation(x_train, y_train, z_train, num_folds, p = 5):
	kfold = KFold(n_splits = num_folds)
	len_beta = int((p+1)*(p+2)/2)

	R2_scores_cv = np.zeros(num_folds)
	MSE_scores_cv = np.zeros(num_folds)
	beta_array = np.zeros( (num_folds, len_beta) )
	R2_sum = 0
	MSE_sum = 0
	i = 0
	for train_inds, test_inds in kfold.split(x_train):
		x_train_k = x_train[train_inds]
		y_train_k = y_train[train_inds]
		z_train_k = z_train[train_inds]
		z_train_1d = np.ravel(z_train_k)

		x_test_k = x_train[test_inds]
		y_test_k = y_train[test_inds]
		z_test_k = z_train[test_inds]
		z_test_1d = np.ravel(z_test_k)

		# Compute model with train data from fold
		X_k = CreateDesignMatrix_X(x_train_k, y_train_k, p)
		beta_k = np.linalg.inv( np.dot(X_k.T, X_k) ) .dot(X_k.T) .dot(z_train_1d)
		beta_array[i] = beta_k

		# Predict with trained model using test data from fold
		X_test_k = CreateDesignMatrix_X(x_test_k, y_test_k, p)
		z_pred_k = X_test_k @ beta_k

		# Compute R2 and MSE scoores
		R2_sum += metrics.r2_score(z_test_1d, z_pred_k)
		MSE_sum += metrics.mean_squared_error(z_test_1d, z_pred_k)
		i += 1

	# Return mean of all MSEs for all folds
	return R2_sum/num_folds, MSE_sum/num_folds

# Create design matrix, find beta and predict
def predict_poly(x, y, z, p):
	X = CreateDesignMatrix_X(x, y, p)
	z_ = np.ravel(z)
	beta = np.linalg.inv(np.dot(X.T, X)) .dot(X.T) .dot(z_)
	z_pred = X @ beta

	return z_, z_pred
