class LinearRegression:
    """A simple linear regression model implemented from scratch."""

    def __init__(self):
        self.coef_ = 0.0  # slope
        self.intercept_ = 0.0  # intercept

    def fit(self, X, y, lr=0.01, epochs=1000):
        """Fit the linear regression model using gradient descent.

        Args:
            X (list or tuple): 1D iterable of feature values.
            y (list or tuple): 1D iterable of target values.
            lr (float): Learning rate.
            epochs (int): Number of gradient descent iterations.
        """
        n = len(X)
        # Initialize parameters
        w = 0.0
        b = 0.0

        for _ in range(epochs):
            # Compute predictions
            y_pred = [w * xi + b for xi in X]
            # Compute gradients
            dw = (-2 / n) * sum((yi - y_pred_i) * xi for xi, yi, y_pred_i in zip(X, y, y_pred))
            db = (-2 / n) * sum(yi - y_pred_i for yi, y_pred_i in zip(y, y_pred))
            # Update parameters
            w -= lr * dw
            b -= lr * db

        self.coef_ = w
        self.intercept_ = b

    def predict(self, X):
        """Predict target values for given feature values.

        Args:
            X (list or tuple): 1D iterable of feature values.

        Returns:
            list: Predicted target values.
        """
        return [self.coef_ * xi + self.intercept_ for xi in X]
