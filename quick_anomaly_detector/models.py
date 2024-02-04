import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#########################################
#   Gaussian Based Anomaly Detection    #
#########################################
# sphinx-apidoc -o docs quick_anomaly_detector 
# select epsilon base on F1
class AnomalyDetectionModel:
    """
    Anomaly Detection Model using Gaussian Distribution.

    This class provides a simple implementation of an anomaly detection model
    based on the Gaussian distribution. It includes methods for estimating
    Gaussian parameters, calculating p-values, selecting the threshold, and making predictions.

    Attributes:
        :ivar mu_train: Mean vector of the training data.
        :vartype mu_train: ndarray

        :ivar var_train: Variance vector of the training data.
        :vartype var_train: ndarray

        :ivar p_values_train: P-values for training data.
        :vartype p_values_train: ndarray

        :ivar p_values_val: P-values for validation data.
        :vartype p_values_val: ndarray

        :ivar epsilon: Chosen threshold for anomaly detection.
        :vartype epsilon: float

        :ivar f1: F1 score corresponding to the chosen threshold.
        :vartype f1: float

    .. code-block:: python

        from quick_anomaly_detector.models import AnomalyDetectionModel

        # Load your datasets (X_train, X_val, y_val)
        # ...

        # Create an instance of AnomalyDetectionModel
        model = AnomalyDetectionModel()

        # Train the model
        model.train(X_train, X_val, y_val)

        # Predict anomalies in the validation dataset
        anomalies = model.predict(X_val)

        print(anomalies)

    .. note::
        The anomaly detection model assumes that the input data follows a Gaussian distribution.

    .. warning::
        This class is designed for educational purposes and may not be suitable for all types of data.
    """

    def __init__(self):
        """
        Initialize the AnomalyDetectionModel.
        """
        self.mu_train = 0
        self.var_train = 0
        self.p_values_train = 0
        self.p_values_val = 0
        self.epsilon = 0.05
        self.f1 = 0
    
    def estimate_gaussian(self, X):
        m, n = X.shape
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        return mu, var
    
    def calculate_p_value(self, X, mu, var):
        mvn = multivariate_normal(mean=mu, cov=np.diag(var))
        p_values = mvn.pdf(X)
        return p_values
    
    def select_threshold(self, y_val, p_val): 
        best_epsilon = 0
        best_F1 = 0
        F1 = 0
        step_size = (max(p_val) - min(p_val)) / 1000
        for epsilon in np.arange(min(p_val), max(p_val), step_size):
            predictions = (p_val < epsilon).astype(int)
            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon
        return best_epsilon, best_F1
    

    def train(self, X_train, X_val, y_val):
        """
        Train the AnomalyDetectionModel.

        :param X_train: Training data matrix.
        :type X_train: ndarray

        :param X_val: Validation data matrix.
        :type X_val: ndarray

        :param y_val: Ground truth labels for validation data.
        :type y_val: ndarray
        """
        self.mu_train, self.var_train = self.estimate_gaussian(X_train)
        self.p_values_train = self.calculate_p_value(X_train, self.mu_train, self.var_train)
        self.p_values_val = self.calculate_p_value(X_val, self.mu_train, self.var_train)
        self.epsilon, self.f1 = self.select_threshold(y_val, self.p_values_val)

    def predict(self, X):
        """
        Predict outliers in the input data.

        :param X: Data matrix for prediction.
        :type X: ndarray

        :return: Boolean array indicating outliers.
        :rtype: ndarray
        """
        p_values = self.calculate_p_value(X, self.mu_train, self.var_train)
        outliers = p_values < self.epsilon
        return outliers
