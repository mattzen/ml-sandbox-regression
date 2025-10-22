from linear_regression import LinearRegression

def generate_data(n=20, slope=2.0, intercept=1.0, noise=0.0):
    """Generate synthetic linear data with optional Gaussian noise."""
    import random
    X = []
    y = []
    for _ in range(n):
        x_val = random.uniform(-10, 10)
        X.append(x_val)
        noise_val = random.gauss(0, noise)
        y.append(slope * x_val + intercept + noise_val)
    return X, y

def test_fit_exact_fit():
    """Test whether LinearRegression exactly fits linearly related data without noise."""
    X, y = generate_data(n=50, slope=3.5, intercept=-1.2, noise=0.0)
    model = LinearRegression(learning_rate=0.01, n_iterations=10000)
    model.fit(X, y)
    assert abs(model.slope - 3.5) < 1e-3, f"Expected slope ~ 3.5, got {model.slope}"
    assert abs(model.intercept + 1.2) < 1e-3, f"Expected intercept ~ -1.2, got {model.intercept}"
    print("Test exact fit passed.")

def test_predict_new_point():
    """Test model predictions on new data points after fitting."""
    X, y = generate_data(n=30, slope=2.0, intercept=5.0, noise=0.0)
    model = LinearRegression(learning_rate=0.01, n_iterations=10000)
    model.fit(X, y)
    x_new = 7.5
    y_expected = 2.0 * x_new + 5.0
    y_pred = model.predict([x_new])[0]
    assert abs(y_pred - y_expected) < 1e-2, f"Prediction error too high: {y_pred} vs {y_expected}"
    print("Test predict new point passed.")


def main():
    test_fit_exact_fit()
    test_predict_new_point()

if __name__ == "__main__":
    main()
