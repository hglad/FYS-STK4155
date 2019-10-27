from functions import *

def main():
    dataset = int(sys.argv[1])
    X, y = load_dataset(dataset)


    # Split into train and test data
    print (X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # logreg_sklearn(X_train, X_test, y_train, y_test)
    my_logreg(X_train, X_test, y_train, y_test)


def main_NN():
    dataset = int(sys.argv[1])
    X, y = load_dataset(dataset)
    n, m = X.shape
    print (y.shape)

    NN = NeuralNet(X, y, 1, int(m/2), 2)
    iters = 10000
    gamma = 1e-2

    # Training loop
    for i in range(iters):
        NN.feed_forward()
        NN.back_propagation(gamma)

    y_pred = NN.predict()

    # Create confusion matrix and visualize it
    accuracy = np.mean(y_pred == y[:,0])


    print (accuracy)
    conf_matrix = metrics.confusion_matrix(y, y_pred)
    sb.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()



if __name__ == '__main__':
    main_NN()

# dataset 1: 100 % accuracy, random_state=123, test_size=0.3
# iters = 50000
# gamma = 5e-2
# beta_0 = np.random.uniform(-10000,10000,m)         # random initial weights

# dataset 0:
# Accuracy: 0.824912 (gamma = 1e-06, 500000 iters)
# Correctly classified: 7053
# Default classified as non-default: 1175
# Non-default classified as default: 322





#
