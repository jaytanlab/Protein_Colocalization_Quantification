import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the logistic function
def sigmoid(x, A, x0, k):
    return A / (1 + np.exp(-(x - x0) / k))

# Your provided x and y data
xdata = np.log(np.array([1e-10, 0.1, 1.0, 10.0]))
ydata = np.array([352.3333333333333, 301.9, 234.55555555555554, 109.33333333333333])
ydata2 = np.array([328.5555556, 278.3333333, 298.5555556, 317])
ydata = ydata / ydata.max() * 100
ydata2 = ydata2 / ydata2.max() * 100

# Fit the data to the logistic function
params, covariance = curve_fit(sigmoid, xdata, ydata, p0=[400, 2, -1], maxfev=100000)
params2, covariance2 = curve_fit(sigmoid, xdata, ydata2, p0=[400, 2, -1], maxfev=100000)

# Generate x values for plotting the curve
x_values = np.linspace(min(xdata), 20, 1000)
y_values = sigmoid(x_values, *params)
y_values2 = sigmoid(x_values, *params2)

# Plotting
plt.scatter(xdata, ydata, color='red', label='-C53 Data')
plt.scatter(xdata, ydata2, color='blue', label='+C53 Data')
plt.plot(x_values, y_values, label='-C53 S-curve')
plt.plot(x_values, y_values2, label='+C53 S-curve')
plt.legend()
plt.grid(True)
plt.show()
