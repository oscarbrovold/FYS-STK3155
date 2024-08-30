import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def scikit_regression(x, y, degree,task):
    poly = PolynomialFeatures(degree=degree) 
    design_matrix = poly.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(design_matrix, y, test_size=0.2, random_state=42)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    if task == 'b' : plot_training(x, design_matrix, linreg)
    return X_train, X_test, y_train, y_test, linreg

def plot_training(x, design_matrix, linreg):
    plt.plot(x, linreg.predict(design_matrix), color='blue', label='Regression on training data') 

def scatter_testing_data(x, y):
    plt.scatter(x[:,1], y, color='red', label='Test data')

def compute_MSE_test(y_test, ypredict):
    return mean_squared_error(y_test, ypredict)

def compute_MSE_train(y_train, ytilde):
    return mean_squared_error(y_train, ytilde)

def MSE_test_train_complexity(x, y, maxdegree):
    MSE_train, MSE_test = [], []
    for degree in range(1, maxdegree + 1):
        X_train, X_test, y_train, y_test, linreg = scikit_regression(x, y, degree,'c')
        MSE_train.append(compute_MSE_train(y_train, linreg.predict(X_train)))
        MSE_test.append(compute_MSE_test(y_test, linreg.predict(X_test)))
    xplot = np.linspace(1, maxdegree, maxdegree).reshape(-1,1) 

    #Smoothing out the complexity graph (dont know if this is a good idea)
    '''
    poly3 = PolynomialFeatures(degree=3)
    X = poly3.fit_transform(xplot)
    linreg_train = LinearRegression()
    linreg_test = LinearRegression()
    linreg_train.fit(X, MSE_train)
    linreg_test.fit(X, MSE_test)
    plt.plot(xplot, linreg_train.predict(X), color='blue', label='MSE training data')
    plt.plot(xplot, linreg_test.predict(X), color='red', label='MSE testing data')
    plt.xlabel('Model complexity')
    plt.ylabel('Prediction Error')
    '''
    plt.plot(xplot, MSE_train, color='blue', label='MSE training data')
    plt.plot(xplot, MSE_test, color='red', label='MSE testing data')
    
if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    x = np.linspace(-3, 3, n).reshape(-1,1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

    #For task 3b
    '''
    X_train, X_test, y_train, y_test, linreg = scikit_regression(x,y,5,'b')
    scatter_testing_data(X_test, y_test)
    print('MSE for training = {}'.format(compute_MSE_train(y_train, linreg.predict(X_train))))
    print('MSE for test = {}'.format(compute_MSE_test(y_test, linreg.predict(X_test))))
    '''
    #For task 3c
    MSE_test_train_complexity(x, y, 15)
    plt.legend()
    plt.show()
    

