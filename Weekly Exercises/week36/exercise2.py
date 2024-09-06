import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def scale_split(x,y,degree):
    poly = PolynomialFeatures(degree=degree) 
    design_matrix = poly.fit_transform(x)
    scaler_X = StandardScaler(with_std=False)
    scaler_y = StandardScaler(with_std=False)
    design_matrix = scaler_X.fit_transform(design_matrix)
    scaled_y = scaler_y.fit_transform(y)
    design_matrix[:, 0] = 1
    X_train, X_test, y_train, y_test = train_test_split(design_matrix, scaled_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 
    
def OLS(X_train, X_test, y_train, y_test):
    beta_opt = np.linalg.inv(X_train.T@X_train)@X_train.T@y_train
    MSE_train = MSE(y_train, X_train @ beta_opt)
    MSE_test = MSE(y_test, X_test @ beta_opt)
    print('MSE train (OLS) = {:.7f}'.format(MSE_train))
    print('MSE test (OLS) = {:.7f}'.format(MSE_test))

def ridge(X_train, X_test, y_train, y_test, l, degree):
    beta_opt = np.linalg.inv(X_train.T@X_train + l * np.identity(degree + 1))@X_train.T@y_train
    MSE_train = MSE(y_train, X_train @ beta_opt)
    MSE_test = MSE(y_test, X_test @ beta_opt)
    print('\nMSE train (ridge) for lamda: {} = {:.7f}'.format(l, MSE_train))
    print('MSE test (ridge) for lamda: {} = {:.7f}'.format(l, MSE_test))
    return MSE_train, MSE_test

def MSE(y_data, y_predict):
    return mean_squared_error(y_data, y_predict)

def poly5():
    X_train, X_test, y_train, y_test = scale_split(x,y,5)
    lambdas = [10**i for i in range(-4, 1)]
    OLS(X_train, X_test, y_train, y_test)
    for l in lambdas:
        ridge(X_train, X_test, y_train, y_test, l, 5)

def polyN():
    N = [10, 15]
    lambdas = [10**i for i in range(-4, 1)]
    xplot = [10**i for i in range(-4, 1)] 
    MSE_ridge_train_deg10, MSE_ridge_test_deg10 = [], []
    MSE_ridge_train_deg15, MSE_ridge_test_deg15 = [], []
    for n in N:
        print('\nFits for {} degree polynom\n'.format(n))
        X_train, X_test, y_train, y_test = scale_split(x,y,n)
        OLS(X_train, X_test, y_train, y_test)         
        for l in lambdas:
            if n == 10:
                MSE_train, MSE_test = ridge(X_train, X_test, y_train, y_test, l, n)
                MSE_ridge_train_deg10.append(MSE_train)
                MSE_ridge_test_deg10.append(MSE_test)
            if n == 15:
                MSE_train, MSE_test = ridge(X_train, X_test, y_train, y_test, l, n)
                MSE_ridge_train_deg15.append(MSE_train)
                MSE_ridge_test_deg15.append(MSE_test)

    plt.plot(np.log10(xplot), MSE_ridge_test_deg10, linestyle=':', color='blue', label='Polynomial apporximation of degree 10 for testing data')
    plt.plot(np.log10(xplot), MSE_ridge_train_deg10, linestyle=':', color='red', label='Polynomial approximation of degree 10 for training data')
    plt.plot(np.log10(xplot), MSE_ridge_test_deg15, color='green', label='Polynomial apporximation of degree 15 for testing data')
    plt.plot(np.log10(xplot), MSE_ridge_train_deg15, color='yellow', label='Polynomial approximation of degree 15 for training data')
    plt.xlabel('λ-values in log10-base')
    plt.ylabel('MSE-value')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    n = 100
    x = np.linspace(-3, 3, n).reshape(-1,1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
    poly5()
    polyN()

