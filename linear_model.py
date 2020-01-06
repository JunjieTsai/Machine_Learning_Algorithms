import numpy as np
class LinearRegression:
    """ Ordinary least squares Linear Regression Using Normal Equation,
    This is the Closed Form Solution for OLS.
    
    ----------
    Parameters
    ----------
    fit_intercept: bool, optional, default True
    Whether to calculate the intercept for this model.
    
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    reg = LinearRegression()
    reg = reg.fit(X_train, Y_train)
    print(reg.theta)
    reg.predict(X_test)
    
    """
    
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.theta = None
        
    def fit(self, X, Y):
        if self.fit_intercept: # add intercept: add X0 = 1 to each constance
            X = np.c_[np.ones((len(X),1)), X] 
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return self
    
    def predict(self, X):
        if self.fit_intercept: # add intercept: add X0 = 1 to each constance
            X = np.c_[np.ones((len(X),1)), X]
        return np.dot(X, self.theta)
    
class RidgeRegression(LinearRegression):
    """ Ridge Regression Using Normal Equation,
    This is the Closed Form Solution for Ridge.
    
    ----------
    Parameters
    ----------
    fit_intercept: bool, optional, default True
    Whether to calculate the intercept for this model.
    
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    reg = RidgeRegression(1)
    reg = reg.fit(X_train, Y_train)
    print(reg.theta)
    reg.predict(X_test)

    """
    
    def __init__(self, alpha = 1, fit_intercept = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.theta = None
        
    def fit(self, X, Y):
        if self.fit_intercept: # add intercept: add X0 = 1 to each constance
            X = np.c_[np.ones((len(X),1)), X] 
        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * np.eye(X.shape[1])).dot(X.T).dot(Y)
        return self