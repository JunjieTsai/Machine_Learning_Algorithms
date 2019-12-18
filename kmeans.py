import math
import numpy as np
import random
from itertools import repeat
    
def euclidean_distance(point1, point2) -> float:
    return math.sqrt(sum(map(lambda x: math.pow((x[0]-x[1]), 2), zip(point1, point2))))

class kmeans(object):
    """ K-Means clustering.
    ----------
    Parameters
    ----------
    n_clusters : int, optional, default: 8
    The number of clusters to form as well as the number of centroids to generate.  
    
    --------
    Examples
    --------
    import numpy as np
    from sklearn.datasets import load_iris
    iris= load_iris()
    X = iris.data    
    model = kmeans(3, random_state=0)
    model = model.fit(X)
    labels = np.array(model.labels)
    print(labels)

    """
    
    def __init__(self, k, max_iter = 10, random_state = 0):
        self.k = k
        self.max_iter = 10
        self.random_state = 0        
        self.X = None
        self.labels = None
        self.centroids = None
        
    def fit(self, X):
        """ Compute k-means clustering.
        ----------
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        self.X = X
        self.centroids = self._init_centroids()        
        old_centroids = None
        n_iter = 0
        while n_iter <= self.max_iter and not np.array_equal(old_centroids, self.centroids): 
            # stop when reach max_iter or no centroid update 
            n_iter += 1
            old_centroids = np.copy(self.centroids)            
            self.labels = list(map(self._assign_clusters, X)) # E step
            self.centroids = self._update_centroids() # M step
        return self
    
    def _init_centroids(self):
        """ Random Choose K points as initial centroids        
        """
        np.random.seed(self.random_state)
        return self.X[np.random.choice(self.X.shape[0], self.k)]
        
    def _assign_clusters(self, point):
        """E step of the K-means EM algorithm.
        return the index of the nearest centroid as label of an instance
        """        
        return np.argmin(list(map(euclidean_distance, self.centroids, repeat(point))))
        
    def _update_centroids(self):
        """M step of the K-means EM algorithm.
        """
        return list(map(lambda label: np.mean(self.X[list(map(lambda x: x == label, self.labels))], axis=0), sorted(set(self.labels))))
        
    def predict(self, X):
        """ Predict the closest cluster each sample in X belongs to.
        ----------
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        """
        return list(map(self._assign_clusters, X))
