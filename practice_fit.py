import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''# Define the combined function: Gaussian + Rectangular (Boxcar)
def gaussian_boxcar(x, A, mu, sigma, B, x1, x2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    B: Height of Boxcar
    x1: Start of Boxcar
    x2: End of Boxcar
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma**2))
    boxcar = B * ((x >= x1) & (x <= x2)).astype(float)
    return gaussian + boxcar

# Example usage with synthetic data
def example_fit():
    # Generate synthetic data
    x = np.linspace(-10, 10, 500)
    y_true = gaussian_boxcar(x, A=5, mu=0, sigma=1.5, B=4, x1=9, x2=9.8)
    noise = np.random.normal(0, 0.3, x.shape)
    y_noisy = y_true + noise

    # Initial guess for parameters
    initial_guess = [4, 0, 1, 10, 8, 10]

    # Fit the model
    popt, pcov = curve_fit(gaussian_boxcar, x, y_noisy, p0=initial_guess)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_noisy, label='Noisy Data', alpha=0.6)
    plt.plot(x, gaussian_boxcar(x, *popt), label='Fitted Function', linewidth=2)
    plt.plot(x, y_true, label='True Function', linestyle='--')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian + Boxcar Fit')
    plt.grid(True)
    plt.show()

    return popt

# Run the example
params = example_fit()
print("Fitted parameters:", params)'''
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = a[:2]
c = a[8:]
print(b)
print(c)
d = b +c 
#d.extend(b)
#d.extend(c)

print(a)
print(d)
