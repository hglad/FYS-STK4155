from proj1_funcs import *
plt.rcParams.update({'font.size': 12})

np.random.seed(10)
# Make data
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

noise = np.asarray([np.random.normal(0,0.5,n*n)])
noise = np.reshape(noise, (n,n))
z = FrankeFunction(x,y) + noise

x_train, x_test, y_train, y_test, z_train, z_test = mselect.train_test_split( x,y,z , test_size=0.2, shuffle=True )

max_p = 30
polys = range(max_p)
R2_scores = np.zeros(max_p)
MSE_test = np.zeros(max_p)
MSE_train = np.zeros(max_p)
MSE_best_cv = np.zeros(max_p)

error_test = np.zeros(max_p)
bias_test = np.zeros(max_p)
var_test = np.zeros(max_p)

error_train = np.zeros(max_p)
bias_train = np.zeros(max_p)
var_train = np.zeros(max_p)
z_test = np.ravel(z_test)

# Perform cross-validation for different polynomial degrees
for j in range(max_p):
    R2_scores[j], MSE_test[j], MSE_train[j], error_test[j], \
    bias_test[j], var_test[j], error_train[j], \
    bias_train[j], var_train[j] = cross_validation(x, y, z, 5, p=j)

best_poly = polys[np.argmin(MSE_test)]
print (best_poly)

plt.plot(polys, bias_test, '-b', label='bias (test)')
plt.plot(polys, var_test, '-r', label='variance (test)')
plt.plot(polys, MSE_test, '-g', label='MSE (test)')
# plt.axis([0,max_p,0,1])
# plt.title('Prediction on test data')
# plt.legend()
# plt.show()

plt.plot(polys, bias_train, '--b', label='bias (train)')
plt.plot(polys, var_train, '--r', label='variance (train)')
plt.plot(polys, MSE_train, '--g', label='MSE (train)')
# plt.axis([0,max_p,0,1])
plt.title('Prediction on train and test data')
plt.legend()
plt.show()

# Plot quantities as function of polynomial degree
# plt.plot(polys, R2_scores,  '-b', label='R2_cv_test')
plt.plot(polys, MSE_test, '-r', label='MSE_cv_test')
plt.plot(polys, MSE_train, '--r', label='MSE_cv_train')
plt.legend()
plt.show()

# Find MSE using best polynomial (found from CV) on test data from train_test_split
z_test_, z_pred_test = predict_poly(x_test, y_test, z_test, best_poly)

# MSE using train data
z_train_, z_pred_train = predict_poly(x_train, y_train, z_train, best_poly)

print ("MSE (train data):", metrics.mean_squared_error(z_train_, z_pred_train))
print ("MSE (test data):", metrics.mean_squared_error(z_test_, z_pred_test))

# plot_surf(x,y,FrankeFunction(x,y), np.ravel(x_test), np.ravel(y_test), np.ravel(z_test))



#
