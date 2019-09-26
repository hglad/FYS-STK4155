from proj1_funcs import *
plt.rcParams.update({'font.size': 12})

np.random.seed(123)

# Make data
n = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

noise = np.asarray([np.random.normal(0,0.1,n*n)])
noise = np.reshape(noise, (n,n))
z = FrankeFunction(x,y) + noise

# Regression
p = 5
len_beta = int((p+1)*(p+2)/2)
X = CreateDesignMatrix_X(x, y, p)
z_ = np.ravel(z)			# flattened array
beta = np.linalg.inv( np.dot(X.T, X) ) .dot(X.T) .dot(z_)

# Create new points to predict using beta
n_pred = 100
x_pred = np.sort(np.random.rand(n_pred))
y_pred = np.sort(np.random.rand(n_pred))
x_pred, y_pred = np.meshgrid(x_pred,y_pred)

X_pred = CreateDesignMatrix_X(x_pred, y_pred, p)
z_pred_newpoints = X_pred @ beta

# Plot the surface
# plot_surf(x, y, z)
# plot_surf(x_pred, y_pred, z_pred_newpoints)

# Create prediction using training points
z_pred = X @ beta

R2 = metrics.r2_score(z_, z_pred)
MSE = metrics.mean_squared_error(z_, z_pred)
print (R2, MSE)

# Variance of beta
var_beta = np.diag(np.linalg.pinv(np.dot(X.T, X)) * np.var(z_))

# 95% CI of beta
t = 1.96
low_beta = beta - t*np.sqrt(var_beta)
high_beta = beta + t*np.sqrt(var_beta)

plt.plot(range(len_beta), beta)
plt.plot(range(len_beta), low_beta)
plt.plot(range(len_beta), high_beta)
plt.show()



#
