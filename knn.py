import numpy as np
import math

def euclidean_distance(point1: list, point2: list) -> list:
    """ Calculate Euclidean Distance
    """
    return math.sqrt(sum(map(lambda x: math.pow(x[0] - x[1], 2), zip(point1, point2))))

class KNeighborsClassifier:
    """ Classifier based on k-nearest neighbors.
    
    ----------
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    
    --------------------------------
    Algorithm Implementation Summary
    --------------------------------
    - Step 1: Calculate Euclidean Distance.
    - Step 2: Get Nearest Neighbors.
    - Step 3: Make Predictions.
    
    --------
    Examples
    --------
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    iris= load_iris()
    X, Y = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    kNN_classifier = KNeighborsClassifier(n_neighbors=3)
    kNN_classifier = kNN_classifier.fit(X_train,Y_train)
    result = kNN_classifier.predict(X_test)
    """
    
    def __init__(self, n_neighbors=5):
        """
        """
        self.n_neighbors = n_neighbors
        self.X = None
        self.Y = None
    
    def fit(self, X: list, Y: list):
        self.X = X
        self.Y = Y
        return self
        
    def _get_neighbors(self, point: list):
        """ Get Nearest Neighbors
        return a list of tuples that represent the K Nearest Neighbors: (x, y, euclidean_distance)
        """
        distances = []
        for x, y in zip(self.X, self.Y):
            distances.append((x, y, euclidean_distance(x, point)))
        distances.sort(key=lambda x: x[2])
        return distances[:self.n_neighbors]
    
    def predict(self, X: list):
        """ predict the classified label as the most label that the k nearest neighbors have.
        """
        predict_labels = []
        for i in X:
            labels = list(map(lambda x: x[1], self._get_neighbors(i)))
            predict_label = np.argmax(np.bincount(labels))
            predict_labels.append(predict_label)
        return predict_labels

class KNeighborsRegressor(KNeighborsClassifier):
    """Regression based on k-nearest neighbors.
    
    ----------
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    
    boston = load_boston()
    X, Y = boston.data, boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    kNN_regressor = KNeighborsRegressor(n_neighbors=3)
    kNN_regressor = kNN_regressor.fit(X_train,Y_train)
    result = kNN_regressor.predict(X_test)
    """
    
    def predict(self, X: list):
        """ predict the value as the average of the k nearest neighbors.
        """
        predict_values = []
        for i in X:
            values = list(map(lambda x: x[1], self._get_neighbors(i)))
            predict_value = np.mean(values)
            predict_values.append(predict_value)
        return predict_values