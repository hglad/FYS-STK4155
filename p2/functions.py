import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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


def prob(x, weights):
    t = np.dot(x, weights)          # weighted quantities
    sg = 1./(1 + np.exp(-t))        # sigmoid
    return sg


def my_logreg(X_train, X_test, y_train, y_test):
    np.random.seed(1)
    success = [1 for i in y_train if i==1]
    fail    = [0 for i in y_train if i==0]

    # X_train[samples, features]
    n = len(X_train[:,0])                    # number of training samples
    m = len(X_train[0,:])                    # number of features

    weights = np.random.uniform(-0.0001,0.0001,m) # random initial weights
    p = prob(X_train, weights)        # weighted probabilities

    tot_cost = -(1./n)*np.sum( y_train*np.log(p) + (1 - y_train)*np.log(1 - np.log(p)))
    print (tot_cost)

    # Predict using random weights
    predict = prob(X_test, weights)
    y_pred = (predict >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)
    diff = y_test-y_pred

    print ("Accuracy:", accuracy)
    print ("Correctly classified:", np.sum(diff==0))
    print ("Default classified as non-default:", np.sum(diff==1))
    print ("Non-default classified as default:", np.sum(diff==-1))

    # Confusion matrix of predicted vs. true classes
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()





#
