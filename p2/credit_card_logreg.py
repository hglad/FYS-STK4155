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
    from sklearn.neural_network import MLPClassifier
    dataset = int(sys.argv[1])
    X, y = load_dataset(dataset)
    n, m = X.shape
    # print (y.shape)

    iters = 10000
    gamma = 1e-3
    print (iters, gamma)
    # params = np.logspace(np.log10(1e-1), np.log10(1e-8), 8)

    neuron_lengths_h1 = np.arange(1, 20)
    neuron_lengths_h2 = np.arange(0, 3)
    neuron_lengths_h3 = np.arange(0, 1)

    # layer_lengths = [3,4,3]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

    best_accuracy = 0
    accuracy_scores = np.zeros(len(neuron_lengths_h1)*len(neuron_lengths_h2)*len(neuron_lengths_h3))

    NN = NeuralNet(X_train, y_train, neuron_lengths=[10], n_categories=2)
    NN.train(iters, gamma)
    y_pred = NN.predict(X_train)
    accuracy = np.mean(y_pred == y_train[:,0])
    ConfMatrix(y_train[:,0], y_pred)

    print ("gamma =", gamma)
    print ("accuracy =", accuracy)
    print ("--------------\n")
    exit()
    config = 0
    best_neurons = []
    for i in range(len(neuron_lengths_h1)):
        for j in range(len(neuron_lengths_h2)):
            for k in range(len(neuron_lengths_h3)):
                neurons_per_layer = [neuron_lengths_h1[i], neuron_lengths_h2[j], neuron_lengths_h3[k]]
                # neurons_per_layer = [neuron_lengths_h1[i]]
                # print (neurons_per_layer)
                NN = NeuralNet(X_train[0:3], y_train[0:3], neuron_lengths=neurons_per_layer, n_categories=2)
                NN.train(iters, gamma)
                # exit()
                y_pred = NN.predict(X_train[0:3])
                print ("gamma =", gamma)
                accuracy = np.mean(y_pred == y_train[0:3])
                # print (y_pred.shape, y_train.shape)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_neurons = neurons_per_layer
                    # best_neurons = [neuron_lengths_h1[i]]

                accuracy_scores[config] = accuracy
                print ("accuracy =", accuracy, "best =", best_accuracy, best_neurons)
                print ("--------------\n")
                # ConfMatrix(y_train[:,0], y_pred)
                config += 1

# [21,5,4]

    plt.hist(accuracy_scores)
    plt.show()
        # scikit-learn NN

        # scikit_NN = MLPClassifier(solver='lbfgs', alpha=0, learning_rate='constant', learning_rate_init=gamma, activation='logistic', hidden_layer_sizes=int(m), random_state=1,max_iter=iters)
        #
        # scikit_NN.fit(X, y[:,0])
        # y_pred = scikit_NN.predict(X)
        # ConfMatrix(y[:,0], y_pred)

    print ('Best accuracy')
    print (best_accuracy, best_neurons, "neurons")

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


"""NN"""
# using dataset 2
# iters = 10000
# gamma = 1e-5
# NN = NeuralNet(X, y, n_h_layers=4, n_h_neurons=6, n_categories=2)
# -> 0.927 accuracy
#
# gamma = 1e-05
# ----Neural network----
# 30 input values
# 1 hidden layers
# 1 neurons per hidden layer
# 2 output categories
#
# iters = 100000
# accuracy 0.9806678383128296


# testing for 1-10 hidden layers, 1-11 neurons per layer
# -> 1 layer 5 neurons best, 0.9876977152899824 accuracy



#
