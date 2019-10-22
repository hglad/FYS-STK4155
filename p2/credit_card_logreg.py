from functions import *

def main():
    # Read file and create dataframe
    df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values={})
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    # Create matrix X of explanatory variables (23 features)
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    # target variable: if customer defaults or not
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
    y = y[:,0]

    print (df.head())
    ncols = X.shape[1]

    # Categorical variables to one-hots
    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
        [("", onehotencoder, [3]),],
        remainder="passthrough"
    ).fit_transform(X)

    print (X)
    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # logreg_sklearn(X_train, X_test, y_train, y_test)
    my_logreg(X_train, X_test, y_train, y_test)

def main2():
    infile = open('marks.txt', 'r')

    n = 0
    for line in infile:
        n += 1

    X = np.ones((n,3))
    y = np.zeros(n)

    i = 0
    infile = open('marks.txt', 'r')
    for line in infile:
        split = line.split(',')
        X[i,1], X[i,2], y[i] = split[0], split[1], split[2]
        i += 1

    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    my_logreg(X_train, X_test, y_train, y_test)
    # logreg_sklearn(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main2()





#
