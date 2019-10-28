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

def load_dataset(dataset):
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    scaler = StandardScaler(with_mean=False)

    if dataset == 0:
        # Read file and create dataframe
        df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values={})
        df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

        df = df.drop(df[(df.BILL_AMT1 == 0)&
                    (df.BILL_AMT2 == 0)&
                    (df.BILL_AMT3 == 0)&
                    (df.BILL_AMT4 == 0)&
                    (df.BILL_AMT5 == 0)&
                    (df.BILL_AMT6 == 0)].index)
        df = df.drop(df[(df.PAY_AMT1 == 0)&
                    (df.PAY_AMT2 == 0)&
                    (df.PAY_AMT3 == 0)&
                    (df.PAY_AMT4 == 0)&
                    (df.PAY_AMT5 == 0)&
                    (df.PAY_AMT6 == 0)].index)

        # Create matrix X of explanatory variables (23 features)
        X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
        # target variable: if customer defaults or not
        y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

        print (df.head())

        # Categorical variables to one-hots
        X = ColumnTransformer(
            [("", onehotencoder, [1,2,3,5,6,7,8,9]),],
            remainder="passthrough"
        ).fit_transform(X)

        X = scaler.fit_transform(X)
        # y_onehot = onehotencoder.fit_transform(y)

    if dataset == 1: # exam marks (towards data science)
        infile = open('marks.txt', 'r')
        n = 0
        for line in infile:
            n += 1

        X = np.ones((n,3))
        y = np.zeros((n,1))

        i = 0
        infile = open('marks.txt', 'r')
        for line in infile:
            l = line.split(',')
            X[i,1], X[i,2], y[i] = l[0], l[1], l[2]
            i += 1

    if dataset == 2: # breast cancer data
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target

        y = np.reshape(y, (len(y), 1))
        X = scaler.fit_transform(X)

    return X, y


def ConfMatrix(y, y_pred):
    # Create confusion matrix and visualize it
    accuracy = np.mean(y_pred == y)
    print (accuracy)
    conf_matrix = metrics.confusion_matrix(y, y_pred)
    sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


class NeuralNet:
    def __init__(self, X, y, n_h_layers, n_h_neurons, n_categories):
        self.X = X
        self.y = y
        self.n_h_layers = n_h_layers
        self.n_h_neurons = n_h_neurons
        self.n_categories = n_categories
        self.n, self.m  = X.shape
        self.iters_done = 0

        self.a = []
        self.w = []
        self.b = []
        # initial weight dimensions different from hidden weights
        w_init = np.random.uniform(-1, 1, (self.m, n_h_neurons))

        self.w.append(w_init)
        # properties of hidden layers
        for l in range(n_h_layers):
            self.b.append(np.random.uniform(-0.01, 0.01,(n_h_neurons)))

        for l in range(n_h_layers-1):
            self.w.append(np.random.uniform(-1, 1, (n_h_neurons, n_h_neurons)))

        self.b.append(np.random.uniform(-0.01, 0.01,(n_categories)))
        self.w.append(np.random.uniform(-1, 1, (n_h_neurons, n_categories)))

        # self.w_output = np.random.uniform(-1, 1, (n_h_neurons, n_categories))
        # self.b_output = np.random.uniform(-0.01, 0.01, n_categories)


    def sigmoid(self, t):
        return 1./(1 + np.exp(-t))

    # @staticmethod
    # @jit
    def feed_forward(self):
        """
        activation in hidden layer: take sigmoid of weighted input,
        a_l = sigmoid( (weights*previous activation) + bias )
        first activation in NN is the input data
        """
        # Calculate first hidden layer activation a_1 using input layer
        # self.z_init = np.matmul(self.X, self.w[0]) + self.b[0]
        # print (self.z_init.shape)
        # a_1 = self.sigmoid(self.z_init)
        #
        # if self.iters_done == 0:
        #     self.a.append(self.X)
        #     self.a.append(a_1)
        # else:
        #     self.a[1] = a_1

        # Input layer
        if self.iters_done == 0:
            self.a.append(self.X)

        # Iterate through hidden layers
        for l in range(1, self.n_h_layers+1):
            # print ('l=', l-1)
            # print (self.a[l-1].shape, self.w[l-1].shape, self.b[l-1].shape)
            self.z = np.matmul(self.a[l-1], self.w[l-1]) + self.b[l-1]

            if self.iters_done == 0:
                self.a.append( self.sigmoid(self.z))
            else:
                self.a[l] = self.sigmoid(self.z)

        # Output layer
        # print (self.a[-1].shape, self.w[-1].shape, self.b[-1].shape)
        self.z_output = np.matmul(self.a[-2], self.w[-1]) + self.b[-1]
        self.a_output = self.sigmoid(self.z_output)     # final probabilities

        if self.iters_done == 0:
            self.a.append(self.a_output)
        else:
            self.a[-1] = self.a_output

    def back_propagation(self):
        delta_L = self.a[-1] - self.y   # error in output layer
        self.b_output_grad = np.sum(delta_L, axis=0)
        self.b_grad = self.b_output_grad

        for l in range(self.n_h_layers+2, 1, -1):
            # print (l-2)
            # print (error_output.shape, self.w[l-2].T.shape, self.a[l-2].shape)
            # print (self.a[l-1].shape, self.w[l-1].shape, self.b[l-1].shape)
            # Gradients
            if (l == self.n_h_layers+2):
                self.w_grad = np.matmul(self.a[l-2].T, delta_L)
                self.b_grad = np.sum(delta_L, axis=0)

            else:
                # print (l-1)
                # print (self.a[l-1].shape, self.w[l-2].shape)
                error_hidden = np.matmul(delta_L, self.w[-1].T) * self.a[l-1] * (1 - self.a[l-1])
                # print (error_output.shape)
                self.b_grad = np.sum(error_hidden, axis=0)
                self.w_grad = np.matmul(self.a[l-2].T, error_hidden)

            if self.lmbd > 0.0:
                self.w_output_grad += self.lmbd * self.w[l-2]
                self.w_grad += self.lmbd * self.w[l-2]

            # Optimize weights/biases
            # print(self.w[l-2].shape, self.w_grad.shape, self.b[l-2].shape, self.b_grad.shape)
            self.w[l-2] -= self.gamma * self.w_grad
            self.b[l-2] -= self.gamma * self.b_grad

        self.iters_done += 1
        print ("iter:",self.iters_done)

    def fit(self, iters=10000, gamma=1e-3,lmbd=0):
        """
        Perform feed-forward and back propagation for given iterations
        """
        self.gamma = gamma
        self.lmbd = lmbd
        for i in range(iters):
            self.feed_forward()
            self.back_propagation()

        self.a = self.a[-1]
        print (self.a.shape)


    # Perform classification from a_output:
    def predict(self, y):
        # y_pred = [i for i in self.a_output[i,:]] # convert to 0 or 1
        y_pred = np.zeros(self.n)
        for i in range(self.n):
            ind = np.argmax(self.a_output[i,:])
            y_pred[i] = int(ind)

        accuracy = np.mean(y_pred == y)
        # print (accuracy)

        return y_pred
        #return np.argmax(self.a_output, axis=1)


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
