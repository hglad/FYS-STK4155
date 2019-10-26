import numpy as np
import pandas as pd
import seaborn as sb
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import fmin_tnc

np.random.seed(1)

class NeuralNet:
    def __init__(self, X, y, n_h_layers, n_h_neurons, n_categories):
        self.X = X
        self.y = y
        self.n_h_layers = n_h_layers
        self.n_h_neurons = n_h_neurons
        self.n_categories = n_categories
        self.n, self.m  = X.shape

        # w[layers, features, neurons]
        self.w = np.random.uniform(-1, 1, (self.m, n_h_neurons))
        self.b = np.random.uniform(-0.01, 0.01, self.m)
        self.z = np.zeros(n_h_neurons)
        self.a = np.zeros(n_h_neurons)

        self.output_w = np.random.uniform(-1, 1, (n_categories, n_h_neurons))
        self.output_b = np.random.uniform(-0.01, 0.01, n_categories)
        self.output_z = np.zeros(n_categories)
        self.output_a = np.zeros(n_categories)


    def sigmoid(self, t):
        return 1./(1 + np.exp(-t))


    def feed_forward(self):
        # Hidden layer
        for j in range(self.n_h_neurons):
            self.z[j] = np.sum(self.w[:,j]*self.X[j,:] + self.b[j])
            self.a[j] = self.sigmoid(self.z[j])

        # Output layer
        for j in range(self.n_categories):
            print (j)
            self.output_z[j] = np.sum(self.output_w[j,:]*self.a + self.output_b[j])
            self.output_a[j] = np.exp(self.output_z[j]) / np.sum(np.exp(self.output_z[j]))

    def predict(self):
        iters = 100000
        gamma = 1e-4
        opt_w, norm = gradient_descent(self.X, self.w.T, self.y, iters, gamma)
        # for i in range(self.n):
        #     self.z[0,i] = self.w[0,i]*self.X[i,:] + self.b[0,i]

        pred = np.dot(self.X, opt_w) + self.b
            # pred = self.sigmoid(temp)

        # Convert to binary values, compute accuracy
        y_pred = (pred >= 0.5).astype(int) # convert to 0 or 1
        accuracy = np.mean(y_pred == self.y)
        print (accuracy)

        return y_pred


def logreg_sklearn(X_train, X_test, y_train, y_test):
    # Create regressor
    logreg = LogisticRegression()

    # Perform fit with train data and prediction on test data
    logreg.fit(X_train, y_train.ravel())
    y_pred = logreg.predict(X_test)
    print (logreg.score(X_test, y_test))

    # Print metrics of model used on test data
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

    # Create confusion matrix and visualize it
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()

    return


def total_cost(n, y, p, beta):
    # tot_cost = -(1./n)*np.sum( y_train*np.log(p) + (1 - y_train)*np.log(1 - np.log(p)))
    tot_cost = -np.sum(y*beta - np.log(1 + np.exp(y*beta)))
    return tot_cost


def prob(x, beta):
    t = np.dot(x, beta)          # weighted quantities
    sg = 1./(1 + np.exp(-t))     # sigmoid
    return sg


def gradient(m, x, y, beta):
    p = 1./(1 + np.exp(-x@beta))        # activation function
    return np.dot(x.T, (p - y))


def gradient_descent(x, beta, y, iters=100, gamma=1e-2):
    # Standard gradient descent
    m = x.shape[1]
    gamma_0 = gamma
    for i in range(iters):
        grad = gradient(m, x, y, beta)
        new_beta = beta - grad*gamma

        norm = np.linalg.norm(new_beta - beta)

        # if (abs(norm) < 1e-12):
        #     gamma *= 1.01
        # if (abs(norm) > 0):
        #     gamma *= 0.99

        beta = new_beta
        # print (norm)
        if (norm < 1e-10):
            return beta, norm

    print (norm, gamma)
    return beta, norm


def my_logreg(X_train, X_test, y_train, y_test):
    # success = [1 for i in y_train if i==1]
    # fail    = [0 for i in y_train if i==0]

    # X[samples, features]
    n = X_train.shape[0]                    # number of training samples
    m = X_train.shape[1]                    # number of features
    # m = len(X[0])
    iters = 500000
    # gamma = 1e-8
    # beta_0 = np.random.uniform(-10000,10000,m)         # random initial weights
    # opt_beta, norm = gradient_descent(X_train, beta_0, y_train, iters, gamma)

    params = np.logspace(np.log10(1e-6), np.log10(1e0), 7)

    for gamma in params:
        beta_0 = np.random.uniform(-10,10,m)
        beta_0 = np.reshape(beta_0, (m,1))
        opt_beta, norm = gradient_descent(X_train, beta_0, y_train, iters=iters, gamma=gamma)

        # Predict using optimal weights
        # print ("Initial beta:", beta_0)
        # print ("Optimal beta:", opt_beta)

        predict = prob(X_test, opt_beta)      # values between 0 and 1
        y_pred = (predict >= 0.5).astype(int) # convert to 0 or 1
        accuracy = np.mean(y_pred == y_test)
        diff = y_test - y_pred

        print ("Accuracy: %g (gamma = %g, %d iters)" % (accuracy, gamma, iters))
        print ("Correctly classified:", np.sum(diff==0))
        print ("Default classified as non-default:", np.sum(diff==1))
        print ("Non-default classified as default:", np.sum(diff==-1))
        print ("")

        # Confusion matrix of predicted vs. true classes
        # conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        # sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        # plt.title('Confusion matrix (default = 1)')
        # plt.ylabel('True value')
        # plt.xlabel('Predicted value')
        # plt.show()

        # plt.plot(opt_beta, '-ro')
        # plt.show()
        #
        # plt.plot(y_pred, '-bo')
        # plt.show()






#
