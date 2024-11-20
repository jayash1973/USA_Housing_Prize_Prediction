import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model using Normal Equation
        """
        try:
            # Add column of ones for intercept
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            
            # Calculate theta using normal equation with regularization
            theta = np.linalg.inv(X_b.T.dot(X_b) + 1e-8 * np.eye(X_b.shape[1])).dot(X_b.T).dot(y)
            
            self.intercept = float(theta[0])
            self.coefficients = theta[1:].astype(float)
            
            print("Model fitted successfully!")
            print(f"Intercept: {self.intercept}")
            print(f"Number of coefficients: {len(self.coefficients)}")
            
            return self
            
        except Exception as e:
            print(f"Error in fitting model: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet!")
        return np.dot(X, self.coefficients) + self.intercept