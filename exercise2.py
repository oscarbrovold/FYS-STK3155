import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def initiate_func():
    x = np.random.rand(100,1)
    y = 2.0 + 5 * x**2 + 0.1 * np.random.randn(100,1)
    return x, y

def plot_true_scatterpoints(x, y):
    plt.scatter(x, y, color='red', label='scatter points')

def manual_regression(x, y, plot):
    design_matrix = np.column_stack([x**i for i in range(3)]) # Matrix, 1.st col = x^0, 2. = x^1 ..
    coeffisents = (inv(design_matrix.T@design_matrix))@design_matrix.T@y
    b0, b1, b2 = coeffisents[0], coeffisents[1], coeffisents[2]    
    if plot : plot_manual_reg(b0, b1, b2)

def plot_manual_reg(b0, b1, b2):
    def f(x) : return b0 + b1 * x + b2 * x**2
    x_plot = np.linspace(0, 1, 100)
    plt.plot(x_plot, f(x_plot), color='blue', label='manual regression')

def scikit_regression(x, y, plot):
    poly2 = PolynomialFeatures(degree=2)
    design_matrix = poly2.fit_transform(x)
    linreg = LinearRegression()
    linreg.fit(design_matrix, y)
    if plot : plot_scikit_regression(linreg, poly2)
    return linreg.predict(design_matrix)

def plot_scikit_regression(linreg, poly2):
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    Xplot = poly2.fit_transform(x_plot)
    plt.plot(x_plot, linreg.predict(Xplot), 'y--' , label='Scikit regression')

def compute_MSE(y, ypredict):
    return mean_squared_error(y, ypredict)

def compute_R2(y, ypredict):
    return r2_score(y, ypredict)

if __name__ == '__main__':
    x, y = initiate_func()
    manual_regression(x, y, True)
    ypredict = scikit_regression(x, y, True)
    plot_true_scatterpoints(x, y)
    print('Mean squared error = {:.4f}'.format(compute_MSE(y, ypredict)))
    print('R2 score = {:.4f}'.format(compute_R2(y, ypredict)))
    plt.legend()
    plt.show()