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

        X = np.zeros((n,2))
        y = np.zeros((n,1))

        i = 0
        infile = open('marks.txt', 'r')
        for line in infile:
            l = line.split(',')
            X[i,0], X[i,1], y[i] = l[0], l[1], l[2]
            i += 1

    if dataset == 2: # breast cancer data
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target


        y = np.reshape(y, (len(y), 1))
        # X = scaler.fit_transform(X)

        for i in range(len(X[0,:])):
            X[:,i] = X[:,i]/np.max(X[:,i])

        # print (X)

    print (y.shape)
    return X, y


def ConfMatrix(y, y_pred):
    # Create confusion matrix and visualize it
    conf_matrix = metrics.confusion_matrix(y, y_pred)
    sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


class NeuralNet:
    def __init__(self, X, y, neuron_lengths, n_categories):
        self.X = X
        self.y = y

        # print (y)
        self.n_h_layers = len(neuron_lengths)
        self.n_categories = n_categories
        self.iters_done = 0

        if len(X.shape) > 1:
            self.n_train, self.m_train  = X.shape

        # Special case for a single training point
        else:
            self.X = np.reshape(X, (1, len(X)))
            self.y = np.reshape(y, (1,1))
            self.n_train = 1
            self.m_train = len(X)

        # Do not include layers that have 0 neurons
        self.n_h_neurons = []
        for layer in neuron_lengths:
            if (layer != 0):
                self.n_h_neurons.append(layer)

        self.n_h_layers = len(self.n_h_neurons)

        self.a = []
        self.w = []
        self.b = []
        self.z = []
        print(self.n_train, self.m_train)

        self.print_properties()

        # create structure of activations, weight and bias arrays
        self.create_structure()


    def print_properties(self):
        print ("----Neural network----")
        print (self.m_train, "input values")
        print (self.n_h_layers, "hidden layers")
        print (self.n_h_neurons, "neurons per hidden layer")
        print (self.n_categories, "output categories\n")


    def create_structure(self):
        self.a.append(self.X) # Input layer
        # Input layer -> first hidden layer weights
        self.w.append(np.random.uniform(-1, 1, (self.m_train, self.n_h_neurons[0])))

        # Hidden layers
        for l in range(self.n_h_layers):
            self.b.append(np.random.uniform(-0.01, 0.01,(self.n_h_neurons[l])))
            self.a.append(np.zeros(self.n_h_neurons[l]))
            self.z.append(np.zeros(self.n_h_neurons[l]))

        for l in range(self.n_h_layers-1):
            self.w.append(np.random.uniform(-1, 1, (self.n_h_neurons[l], self.n_h_neurons[l+1])))

        self.b.append(np.random.uniform(-0.01, 0.01,(self.n_categories)))
        self.w.append(np.random.uniform(-1, 1, (self.n_h_neurons[-1], self.n_categories)))
        self.a.append(np.zeros(self.n_categories))  # Output layer
        self.z.append(np.zeros(self.n_categories))


    def activation(self, x, func='sigmoid'):

        if (func == 'sigmoid'):
            t = 1./(1 + np.exp(-x))

        if (func == 'softmax'):
            # t = np.zeros((x.shape))
            # softmax = np.zeros((x.shape[1]))
            # for i in range(x.shape[0]):
            #     softmax = np.exp(x[i,:])/np.sum(np.exp(x[i,:]))
            #     t[i] = softmax

            exp_term = np.exp(x)
            t = exp_term / np.sum(exp_term, axis=1, keepdims=True)

            # if len(x.shape) > 1:
            #     for i in range(self.n_train):
            #         t[i] = np.max(softmax[i,:])
            # print (t)

        return t


    # @staticmethod
    # @jit
    def feed_forward(self):
        """
        activation in hidden layer: take sigmoid of weighted input,
        a_l = sigmoid( (weights*previous activation) + bias )
        first activation in NN is the input data
        """
        # Iterate through hidden layers
        for l in range(1, self.n_h_layers+1):
            self.z[l] = np.matmul(self.a[l-1], self.w[l-1]) + self.b[l-1]
            self.a[l] = np.tanh(self.z[l])
            # print (self.a[l])
            # print (self.a[l], '=', 'tanh(', self.a[l-1], self.w[l-1], '+', self.b[l-1], ')')

        # Output layer
        self.z[-1] = np.matmul(self.a[-2], self.w[-1]) + self.b[-1]

        self.a_output = self.activation(self.z[-1], self.func)
        # print (self.a_output)
        self.a[-1] = self.a_output


    def w_b_gradients(self, delta, l):
        # print (delta, self.a[-1].T, l)
        self.b_grad = np.sum(delta, axis=0)
        self.w_grad = np.matmul(self.a[l].T, delta)
        if self.lmbd > 0.0:
            self.w_grad += self.lmbd * self.w[l]


    def back_propagation(self):
        delta_L = self.a_output - self.y   # error in output layer
        # print (self.a_output)
        total_loss = np.mean(delta_L)
        # print (total_loss)

        old_w = self.w[-1]

        # Output layer

        self.w_b_gradients(delta_L, self.n_h_layers)
        self.w[-1] -= self.gamma * self.w_grad
        self.b[-1] -= self.gamma * self.b_grad

        delta_old = delta_L
        # w_norm = np.linalg.norm(old_w - self.w[-1])
        # print (self.w_grad)
        for l in range(self.n_h_layers, 0, -1):
            # print (l-1)
            # Use previous error to propagate error back to first hidden layer
            delta_h = np.matmul(delta_old, self.w[l].T) * self.a[l] * (1 - self.a[l])
            self.w_b_gradients(delta_h, l-1)

            # Optimize weights/biases
            # print (self.w_grad)
            self.w[l-1] -= self.gamma * self.w_grad
            self.b[l-1] -= self.gamma * self.b_grad

            delta_old = delta_h
            old_w = self.w
            # print (self.w_grad)

        self.iters_done += 1
        sys.stdout.write('iter %d / %d , loss %1.3e \r' % (self.iters_done, self.iters, total_loss))
        sys.stdout.flush()


    def train(self, func='sigmoid', iters=10000, gamma=1e-3, lmbd=0):
        self.func = func
        """
        Perform feed-forward and back propagation for given number of iterations
        """
        self.gamma = gamma
        self.lmbd = lmbd
        self.iters = iters

        for i in range(iters):
            self.feed_forward()
            self.back_propagation()


    def predict(self, X_test, y_test):
        """
        Predict outcome using the trained NN. The activation layers are changed
        according to the input data X.
        """
        if len(X_test.shape) > 1:
            n, m = X_test.shape
        else:
            n = 1
            m = len(X_test)

        self.a[0] = X_test
        self.feed_forward()
        print ("")
        y_pred = (np.zeros(n)).astype(int)

        # Find activation with highest value in output layer
        if len(self.a_output.shape) > 1:
            for i in range(n):
                highest_p = np.argmax(self.a_output[i,:])       # 0 or 1
                # print (self.a_output[i,:])
                y_pred[i] = int(highest_p)
                # print (y_pred[i], self.y[i], type(y_pred[i]))
        else:
            # Special case for a single testing point
            for i in range(n):
                highest_p = np.argmax(self.a_output)       # 0 or 1
                y_pred[i] = int(highest_p)

        return y_pred


    def predict2(self, X_test, y_test):
        self.a[0] = X_test
        self.feed_forward()
        return np.argmax(self.a_output, axis=1)


    def predict_single_output_neuron(self, X_test, y_test):
        self.a[0] = X_test
        self.y_test = y_test
        self.feed_forward()
        y_pred = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):
            # print (self.a_output[i], self.y_test[i,0])
            if self.a_output[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

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
