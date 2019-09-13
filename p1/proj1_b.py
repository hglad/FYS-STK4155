from proj1_funcs import *

# np.random.seed()
# Make data
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

noise = np.asarray([np.random.normal(0,0.1,n*n)])
noise = np.reshape(noise, (n,n))
z = FrankeFunction(x,y) + noise

x_train, x_test, y_train, y_test, z_train, z_test = mselect.train_test_split( x,y,z , test_size=0.2 )

max_p = 20
polys = range(1, max_p+1)
R2_scores = np.zeros(max_p)
MSE_values = np.zeros(max_p)
MSE_best_cv = np.zeros(max_p)

# Perform cross-validation for different polynomial degrees
j = 0
for p in polys:
    R2_scores[j], MSE_values[j] = cross_validation(x_train, y_train, z_train, 10, p=polys[j])
    j += 1

best_poly = polys[np.argmin(MSE_values)]
print (best_poly)

# Plot quantities as function of polynomial degree
plt.plot(polys, R2_scores,  '-b', label='R2_cv_test')
plt.plot(polys, MSE_values, '-r', label='MSE_cv_test')
plt.legend()
plt.show()

# Find MSE using best polynomial (found from CV) on test data from train_test_split
z_test_, z_pred_test = predict_poly(x_test, y_test, z_test, best_poly)

# MSE using train data
z_train_, z_pred_train = predict_poly(x_train, y_train, z_train, best_poly)

print ("MSE (train data):", metrics.mean_squared_error(z_train_, z_pred_train))
print ("MSE (test data):", metrics.mean_squared_error(z_test_, z_pred_test))


#
