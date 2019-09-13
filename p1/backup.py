from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

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
	print (len_beta)
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

# Framework for 3D plotting
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data
n = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
noise = np.asarray([np.random.normal(0,0.1,n*n)])
noise = np.reshape(noise, (n,n))
z += noise

# Customize the z-axis
ax.set_zlim(-0.1, 1.4)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors

# Regression
X = CreateDesignMatrix_X(x, y, p=5)
z_ = np.ravel(z)			# need 1d array for finding beta
# print (X.shape, z.shape)
# print ((np.dot(X.T, X)).shape, z_.shape)
beta = np.linalg.inv( np.dot(X.T, X) ) .dot(X.T) .dot(z_)
print (beta)

# x_pred = np.random.randn(n)/10.
# y_pred = np.random.randn(n)/10.
# pred = np.ones((len(x_pred),len(beta)))

z_pred = X @ beta
# Plot the surface
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
z_pred = np.reshape(z_pred, (20,20))
surf = ax.plot_surface(x, y, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# for i in range(1,n+1):
# 	q = int((i)*(i+1)/2)
# 	for k in range(i+1):
# 		pred[:,q+k] = x_pred**(i-k) * y_pred**k




#
