import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Patricia\PycharmProjects\Linear Regression\Nairobi Office Price Ex.csv')
  # Replace with the actual path

# Extract the necessary columns
x = data['SIZE'].values
y = data['PRICE'].values


# 1. Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 2. Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(y)
    errors = []
    for epoch in range(epochs):
        # Predict y values with current slope and intercept
        y_pred = m * x + c
        # Calculate MSE for the current epoch
        error = mean_squared_error(y, y_pred)
        errors.append(error)

        # Calculate gradients
        dm = (-2 / n) * sum(x * (y - y_pred))
        dc = (-2 / n) * sum(y - y_pred)

        # Update weights
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Print the error for the current epoch
        print(f"Epoch {epoch + 1}, MSE: {error}")

    return m, c, errors


# Initialize parameters
m_initial = np.random.rand()
c_initial = np.random.rand()
learning_rate = 0.0001
epochs = 10

# Train the model
m_final, c_final, errors = gradient_descent(x, y, m_initial, c_initial, learning_rate, epochs)

# Plotting the data points and the line of best fit
plt.scatter(x, y, color="blue", label="Data Points")
y_pred_line = m_final * x + c_final
plt.plot(x, y_pred_line, color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("Linear Regression - Office Price Prediction")
plt.legend()
plt.show()

# Predict the office price for 100 sq. ft.
size_100_pred = m_final * 100 + c_final
print(f"Predicted price for an office of 100 sq. ft.: {size_100_pred}")

plt.savefig('line_of_best_fit.png')
