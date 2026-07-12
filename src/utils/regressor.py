from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

class VectorAutoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_lags=2):
        self.n_lags = n_lags
        self.model = MultiOutputRegressor(LinearRegression())

    def fit(self, X, y):
        # X shape: (num_samples, n_lags*2) - flattened window of positions
        # y shape: (num_samples, 2) - next position vectors
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # X shape: (n_lags, 2) or (num_samples, n_lags*2)
        if X.ndim == 2 and X.shape[0] == self.n_lags and X.shape[1] == 2:
            # Single prediction: flatten
            X_flat = X.flatten().reshape(1, -1)
            prediction = self.model.predict(X_flat)
            return prediction[0]  # shape (2,)
        elif X.ndim == 2 and X.shape[1] == self.n_lags * 2:
            # Batch prediction
            predictions = self.model.predict(X)
            return predictions  # shape (num_samples, 2)
        else:
            raise ValueError(f"Input shape {X.shape} not understood for prediction.")

    @property
    def n_features_in_(self):
        # Return the number of features used in the model
        return self.model.estimators_[0].n_features_in_ if hasattr(self.model, 'estimators_') and self.model.estimators_ else None

    @property
    def coef_(self):
        # Return the coefficients of the underlying regressors
        return [est.coef_ for est in self.model.estimators_] if hasattr(self.model, 'estimators_') else None