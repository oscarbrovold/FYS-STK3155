import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

if __name__ == '__main__':
    # Random seed
    np.random.seed(42)

    # Hyper parameters 
    n = 100
    max_degree = 16
    B = 100

    # function (with noise)
    x = np.linspace(-3,3,n).reshape(-1,1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) 
    noise = np.random.normal(0,0.2,x.shape)
    y_noise = y + noise

    # Bias, variance
    bias = []
    variance = []

    # Design matrix with all columns. If i want degree < max_degree => slice
    X_full = np.column_stack([x**i for i in range(max_degree + 1)])

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_noise, train_size=0.8) 

    for deg in range(max_degree):
        y_pred = np.empty((y_test.shape[0], B)) # Sets up a matrix where each column is y_pred from each bootstrap
        for b in range(B):
            sliced_train_matrix = X_train[:, :deg + 1]
            sliced_test_matrix = X_test[:, :deg + 1]
            X_, y_ = resample(sliced_train_matrix, y_train)
            betas = np.linalg.inv(X_.T @ X_) @ X_.T @ y_ 
            y_pred[:, b] = sliced_test_matrix @ betas.ravel()
        bias.append(np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1, keepdims=True)))

    x_plot = [i for i in range(1, max_degree + 1)]
    plt.plot(x_plot, bias, color='r', label='bias') 
    plt.plot(x_plot, variance, color='b', label='variance')
    plt.legend()
    plt.show()