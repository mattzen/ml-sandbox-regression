# ML Sandbox Regression

This repository contains a minimal machine learning sandbox implementing a simple linear regression algorithm from scratch (no external libraries) and a demonstration on a synthetic dataset.

## Contents

- `linear_regression.py` – Implementation of a simple univariate linear regression model using gradient descent.
- `demo.ipynb` – Jupyter notebook demonstrating how to use the linear regression implementation on a tiny synthetic dataset.
- `test_linear_regression.py` – A minimal test script to verify the correctness of the linear regression implementation.

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/ml-sandbox-regression.git
   cd ml-sandbox-regression
   ```

2. **Run the test script**  
   The test script generates a simple dataset, fits the model, and prints the learned parameters.  
   ```bash
   python test_linear_regression.py
   ```

3. **Open the Jupyter notebook**  
   To explore the demonstration notebook, first install Jupyter Notebook or use JupyterLab (if not already installed). You can then launch the notebook:
   ```bash
   jupyter notebook demo.ipynb
   ```

## Test Script

The `test_linear_regression.py` script creates a simple linear dataset (with optional noise), fits the `LinearRegression` model from `linear_regression.py`, and prints the learned slope and intercept.

## Next Steps

Here are some suggestions for extending this sandbox:
- Expand the implementation to handle multivariate linear regression.
- Add evaluation metrics such as Mean Squared Error (MSE) and R².
- Explore different optimization techniques (e.g., analytical solution).
- Add unit tests with more rigorous scenarios (e.g., edge cases, random data).
