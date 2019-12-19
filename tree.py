import numpy as np
def gini_impurity(y):
    """calculate gini impurity for a split child
    """
    hist = np.bincount(y)
    n = np.sum(hist)
    gini_impurity = 1 - sum([(i/n)**2 for i in hist])
    return gini_impurity

class Node:
    """ Tree Node Object
    """
    def __init__(self, left=None, right=None, split_feature=None, split_threshold=None, gini=0.0, value=None):
        self.left = left # id of the left child of the node
        self.right = right 
        self.split_feature = split_feature # Feature used for splitting the node
        self.split_threshold = split_threshold
        self.gini = gini
        self.value = value

class DecisionTreeClassifier:
    """A decision tree classifier.
    
    ----------
    Parameters
    ----------
    max_depth : int, optional (default=np.inf)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than min_samples_split samples.
        
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        
    ----------
    Attributes
    ----------
    tree : The underlying Tree object. 
    
    n_classes: int
        The number of classes.
    
    n_sample: int
        The number of instances.
    
    n_features: int
        The number of features.
        
    --------
    Examples
    --------
    from sklearn.datasets import load_iris
    iris= load_iris()
    X = iris.data
    Y = iris.target
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    clf.predict(X)
    """
    
    def __init__(self, max_depth = np.inf):        
        self.max_depth = max_depth
        self.tree = None
        
        self.n_features = None
        self.n_sample = None       
        self.n_classes = None
        
    def _split(self, X, Y):
        ''' find and output the optimal split parameters (feature and threshold)
        '''
        costs = [] # Weighted Gini Impurities
        feature_threshold = [] # Tuples: (feature index, threshold)

        # traverse split features
        for i in range(self.n_features):
            values = X[:, i].flatten()
            unique_values = np.unique(values)
            thresholds = (unique_values[:-1] + unique_values[1:])/2 # generate thresholds: midpoints between two adjacent values of the choosen feature

            # traverse split thresholds
            for threshold in thresholds:
                # split X and Y with threshold
                left_index = np.argwhere(values < threshold).flatten()
                right_index = np.argwhere(values >= threshold).flatten()
                left_Y, right_Y = Y[left_index], Y[right_index]
                # Weighted Gini Impurity
                n = len(Y)
                n_left, n_right = len(left_index), len(right_index)
                cost = (n_left / n) * gini_impurity(left_Y) + (n_right / n) * gini_impurity(right_Y)        
                costs.append(cost)
                feature_threshold.append((i, threshold, cost))
                
        return feature_threshold[np.argmin(costs)] # out *feature and *threshold that has min *gini (Weighted Gini Impurity)

    def _grow_tree(self, X, Y, depth=0):
        """ split each node recursively until the maximum depth is reached or arrive homogeneousness
        """
        node = Node()
        node.value = ([np.sum(Y ==i) for i in range(self.n_classes)])
        if depth <= self.max_depth and len(set(Y)) > 1:        
            split_feature, split_threshold, gini = self._split(X, Y) # calculate the optimal split parameters
            print(depth, split_feature, split_threshold)

            indices_left = X[:, split_feature] < split_threshold # indices to split the node data into 2 parts
            X_left, Y_left = X[indices_left], Y[indices_left] # left child data 
            X_right, Y_right = X[~indices_left], Y[~indices_left] # right child data

            node.split_feature = split_feature
            node.split_threshold = split_threshold
            node.left = self._grow_tree(X_left, Y_left, depth + 1) # recursion
            node.right = self._grow_tree(X_right, Y_right, depth + 1) # recursion
            node.gini = gini
        return node

    def fit(self, X, Y):
        """ Build a decision tree fitting the training data
        """
        self.n_features = X.shape[1]   
        self.n_sample = X.shape[0]             
        self.n_classes = len(np.unique(Y))        
        self.tree = self._grow_tree(X, Y)
        return self
        
    def _traverse(self, x, node):
        """ traverse the tree with an instance x to predict y
        """
        if node.split_feature and node.split_threshold:
            if x[node.split_feature] < node.split_threshold:
                node = node.left
                node = self._traverse(x, node)
            else:
                node = node.right
                node = self._traverse(x, node)
        return node

    def predict(self, X):       
        return [np.argmax(self._traverse(x, self.tree).value) for x in X]