import numpy as np

def mse(X, theta, Y):
    return ((np.dot(X, theta) - Y) ** 2).mean()/2

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
    
class Ridge(LinearRegression):
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
    reg = Ridge(1)
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

class BGDRegressor:
    """Linear Regression with Batch Gradient Descent
    
    ----------
    Parameters
    ----------    
    learning_rate: float, default=0.01
    
    max_iter: int, default=1e3
        maximum iterations
    
    tol: float, default=1e-3
        Tolerance, the stopping criterion. 
        Default:     'cost function'
        Alternative: 'step length'. faster, without cost computation.
    
    alpha: float, default=0
        Ridge Regression L2 regularization term
        default: no regularization
        
    fit_intercept: bool, default=True
        Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
    
    random_state: int, default=None
        The seed of the pseudo random number generator to use when initializing theta. 
    
    ----------
    Attributes
    ----------
    theta: ndarray of shape (n_features,)
        Weights assigned to the features.    
    
    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        
    cost: float
        MSE cost of traning data
    
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Linear Regression with Batch Gradient Descent
    reg = BGDRegressor(learning_rate = 0.5, max_iter = 1e6, tol = 1e-3, fit_intercept = True, random_state=42)

    # Ridge Regression with Batch Gradient Descent
    reg = BGDRegressor(learning_rate = 0.5, max_iter = 1e6, tol = 1e-3, alpha = 1e-3, fit_intercept = True, random_state=42)

    reg = reg.fit(X_train, Y_train)
    print('theta:', reg.theta)
    print('iterations:', reg.n_iter)
    print('cost:', reg.cost)
    reg.predict(X_test)
    
    """    
    
    def __init__(self, learning_rate = 0.01, max_iter = 1e3, tol = 1e-3, alpha = 0, fit_intercept = True, random_state = None):
        # Parameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = None
        # Attributes
        self.theta = None
        self.n_iter = None
        self.n_samples = None
        self.n_features = None
        self.cost = None        
        
    def _gradient_descent(self, X, Y):
        np.random.seed(self.random_state)        
        convergence = False
        cost = [np.inf]*2
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
            theta = np.random.randn(self.n_features + 1) # random initialization for theta values, from Gaussian distribution
        else:
            X_b = X
            theta = np.random.randn(self.n_features)
        for iteration in range(int(self.max_iter)):
            gradient = 2/self.n_samples * np.dot(X_b.T, np.dot(X_b, theta) - Y) + self.alpha * theta
            theta = theta - self.learning_rate * gradient
            
            # convergence criteria of 'cost function'
            cost.pop(0)
            cost.append(mse(X_b, theta, Y))            
            # print(cost[1], end='\r')
            
            if np.abs(cost[0] - cost[1]) < self.tol:
                convergence = True
                break
            
            # convergence criteria of 'step length'
            # if np.linalg.norm(gradient) < self.tol: 
                # print("Converge!")
                # break

        if convergence:
            print("Achieve Convergence!")
        else:
            print("Reached Max Iteration! Not Achieve Convergence.")
            
        self.cost = cost[1]
        self.n_iter = iteration
        return theta
    
    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
        self.theta = self._gradient_descent(X, Y)
        return self
    
    def predict(self, X):
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
        else:
            X_b = X
        return np.dot(X_b, self.theta)

class SGDRegressor:
    """Linear Regression with Stochastic Gradient Descent
    
    ----------
    Parameters
    ----------    
    learning_rate: float or string, default=0.01
        Options:
        
        float: [default]
            eta = eta0, 
            constant learning rate
        
        'annealing': 
            simulated annealing with learning schedule, generally faster and more accurate
            please tuning the hyperpameters t0, t1 for better performance

    max_iter: int, default=1e3
        maximum iterations
    
    tol: float, default=1e-3
        Convergence Tolerance, using the stopping criterion of cost function. 

    alpha: float, default=0
        Ridge Regression L2 regularization term
        default: no regularization
        
    fit_intercept: bool, default=True
        Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
    
    random_state: int, default=None
        The seed of the pseudo random number generator to use when initializing theta and selecting data. 
    
    ----------
    Attributes
    ----------
    theta: ndarray of shape (n_features,)
        Weights assigned to the features.    
    
    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        
    cost: float
        MSE cost of traning data
    
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # constant learning rate
    reg = SGDRegressor(learning_rate = 1e-3, max_iter = 1e3, tol = 1e-3, fit_intercept = True, random_state=42)

    # simulated annealing with learning schedule: generally faster and more accurate
    reg = SGDRegressor(learning_rate = 'annealing', max_iter = 1e3, tol = 1e-3, t0 = 1e3/2, t1 = 1e3, fit_intercept = True, random_state=42)

    # Ridge Regression: L2 Regularization 
    reg = SGDRegressor(learning_rate = 'annealing', max_iter = 1e3, tol = 1e-3, t0 = 1e3/2, t1 = 1e3, alpha = 1e-6, fit_intercept = True, random_state=42)

    reg = reg.fit(X_train, Y_train)
    print('theta:', reg.theta)
    print('iterations:', reg.n_iter)
    print('cost:', reg.cost)
    reg.predict(X_test)

    """    
    
    def __init__(self, learning_rate = 0.01, max_iter = 1e3, tol = 1e-3, t0 = 10, t1 = 1000, alpha = 0, fit_intercept = True, random_state = None):
        # Parameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.t0 = t0
        self.t1 = t1
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = None
        # Attributes
        self.theta = None
        self.n_iter = None
        self.n_samples = None
        self.n_features = None
        self.cost = None        
    
    # descent methods
    def _learning_rate(self, i):
        return self.theta - self.learning_rate * self.gradient
    def _learning_schedule(self, t):
        return self.t0 / (t + self.t1)
    def _simulated_annealing(self, i):
        return self.theta - self._learning_schedule(self.n_iter * self.n_samples + i) * self.gradient + self.alpha * self.theta
    
    def _gradient_descent(self, X, Y):        
        # select descent method accoding to the hyperparameter 'learning_rate'
        if type(self.learning_rate) == float:
            self._descent = self._learning_rate
        elif self.learning_rate == 'annealing':
            self._descent = self._simulated_annealing
        else:
            print('learning rate error!')
            
        # initialization
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
            self.theta = np.random.randn(self.n_features + 1) # random initialization for theta values, from Gaussian distribution
        else:
            X_b = X
            self.theta = np.random.randn(self.n_features)
        convergence = False
        np.random.seed(self.random_state)
        
        cost = [np.inf]*2 # queue recoding N cost values
        
        for iteration in range(int(self.max_iter)):
            self.n_iter = iteration
            for i in range(self.n_samples):
                random_index = np.random.randint(self.n_samples)
                Xi = X_b[random_index]
                Yi = Y[random_index]
                self.gradient = 2 / 1 * np.dot(Xi.T, np.dot(Xi, self.theta) - Yi)
                self.theta = self._descent(i)

            # convergence criteria of 'cost function'
            cost.pop(0) 
            cost.append(mse(X_b, self.theta, Y)) # calculate cost each iteration
            print(cost[1], end="\r")
            if np.abs(cost[0] - cost[1]) < self.tol:
                convergence = True
                break
            self.cost = cost[1]
            
        if convergence:
            print("Achieve Convergence!")
        else:
            print("Reached Max Iteration! Not Achieve Convergence.")
            
    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]        
        self._gradient_descent(X, Y)
        return self
        
    def predict(self, X):
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
        else:
            X_b = X
        return np.dot(X_b, self.theta)

class MBGDRegressor:
    """Linear Regression with Mini-Batch Gradient Descent
    
    ----------
    Parameters
    ----------    
    learning_rate: float or string, default=0.01
        Options:
        
        float: [default]
            eta = eta0, 
            constant learning rate
        
        'annealing': 
            simulated annealing with learning schedule, generally faster and more accurate
            please tuning the hyperpameters t0, t1 for better performance

    max_iter: int, default=1e3
        maximum iterations
    
    tol: float, default=1e-3
        Convergence Tolerance, using the stopping criterion of cost function. 
    
    batch_size: int, default=2**5
        the number of instances in each batch
        recommended values: 2**5 ~ 2**10
    
    alpha: float, default=0
        Ridge Regression L2 regularization term
        default: no regularization
        
    fit_intercept: bool, default=True
        Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
    
    random_state: int, default=None
        The seed of the pseudo random number generator to use when initializing theta and selecting data. 
    
    ----------
    Attributes
    ----------
    theta: ndarray of shape (n_features,)
        Weights assigned to the features.    
    
    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        
    cost: float
        MSE cost of traning data
    
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # constant learning rate
    reg = MBGDRegressor(learning_rate = 1e-3, max_iter = 1e3, tol = 1e-3, batch_size = 2**5, alpha = 0, fit_intercept = True, random_state=42)

    # simulated annealing with learning schedule: generally faster and more accurate
    reg = MBGDRegressor(learning_rate = 'annealing', max_iter = 1e3, tol = 1e-3, batch_size = 2**5, t0 = 1e3/2, t1 = 1e3, alpha = 0, fit_intercept = True, random_state=42)

    # Ridge Regression: L2 Regularization 
    reg = MBGDRegressor(learning_rate = 'annealing', max_iter = 1e3, tol = 1e-3, batch_size = 2**5, t0 = 1e3/2, t1 = 1e3, alpha = 1, fit_intercept = True, random_state=42)

    reg = reg.fit(X_train, Y_train)
    print('theta:', reg.theta)
    print('iterations:', reg.n_iter)
    print('cost:', reg.cost)
    reg.predict(X_test)

    """    
    
    def __init__(self, learning_rate = 0.01, max_iter = 1e3, tol = 1e-3, batch_size = 2**5, t0 = 10, t1 = 1000, alpha = 0, fit_intercept = True, random_state = None):
        # Parameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.t0 = t0
        self.t1 = t1
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = None
        # Attributes
        self.theta = None
        self.n_iter = None
        self.n_samples = None
        self.n_features = None
        self.cost = None        
    
    # descent methods
    def _learning_rate(self, i):
        return self.theta - self.learning_rate * self.gradient
    def _learning_schedule(self, t):
        return self.t0 / (t + self.t1)
    def _simulated_annealing(self, i):
        return self.theta - self._learning_schedule(self.n_iter * self.n_samples + i) * self.gradient
    
    def _gradient_descent(self, X, Y):        
        # select descent method accoding to the hyperparameter 'learning_rate'
        if type(self.learning_rate) == float:
            self._descent = self._learning_rate
        elif self.learning_rate == 'annealing':
            self._descent = self._simulated_annealing
        else:
            print('learning rate error!')
            
        # initialization
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
            self.theta = np.random.randn(self.n_features + 1) # random initialization for theta values, from Gaussian distribution
        else:
            X_b = X
            self.theta = np.random.randn(self.n_features)
        convergence = False
        np.random.seed(self.random_state)
        cost = [np.inf]*2 # queue recoding N cost values        
        
        for iteration in range(int(self.max_iter)):
            self.n_iter = iteration
            for i in range(0, self.n_samples, self.batch_size):
                random_indexes = np.random.randint(0, self.n_samples, self.batch_size)
                Xi = X_b[random_indexes]
                Yi = Y[random_indexes]
                self.gradient = 2 / self.batch_size * np.dot(Xi.T, np.dot(Xi, self.theta) - Yi) + self.alpha * self.theta
                self.theta = self._descent(i)

            # convergence criteria of 'cost function'
            cost.pop(0) 
            cost.append(mse(X_b, self.theta, Y)) # calculate cost each iteration
            print(cost[1], end="\r")
            if np.abs(cost[0] - cost[1]) < self.tol:
                convergence = True
                break
            self.cost = cost[1]
            
        if convergence:
            print("Achieve Convergence!")
        else:
            print("Reached Max Iteration! Not Achieve Convergence.")
    
    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]        
        self._gradient_descent(X, Y)
        return self
        
    def predict(self, X):
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
        else:
            X_b = X
        return np.dot(X_b, self.theta)
    
def soft_threshold(rho, alpha):
    if rho < - alpha:
        return rho + alpha
    if rho > alpha:
        return rho - alpha
    else:
        return 0

class Lasso(CDRegressor):
class CDRegressor:
    """Lasso Regression with Coordinate Descent
    
    ----------
    Parameters
    ----------    
    
    max_iter: int, default=1e3
        maximum iterations
    
    tol: float, default=1e-3
        Convergence Tolerance, using the stopping criterion of cost function. 

    alpha: float, default=0.01
        Lasso Regression L1 regularization term
        
    fit_intercept: bool, default=True
        Whether the intercept should be estimated or not. If False, the data is assumed to be already centered.
    
    random_state: int, default=None
        The seed of the pseudo random number generator to use when initializing theta. 
    
    ----------
    Attributes
    ----------
    
    theta: ndarray of shape (n_features,)
        Weights assigned to the features.    
    
    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
    
    cost: float
        MSE cost of traning data
    
    coef_: array
        parameter vector

    intercept_: float
        independent term in decision function
        
    --------
    Examples
    --------
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X, Y = boston.data, boston.target
    X = normalize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # Lasso Regression with Coordinate Descent
    reg = CDRegressor(max_iter = 1e3, tol = 1e-3, alpha = 0.01, fit_intercept = True, random_state=42)

    reg = reg.fit(X_train, Y_train)
    print('theta:', reg.theta)
    print('iterations:', reg.n_iter)
    print('cost:', reg.cost)
    reg.predict(X_test)
    
    """    
    
    def __init__(self, max_iter = 1e3, tol = 1e-3, alpha = 0.01, fit_intercept = True, random_state = None):
        # Parameters
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = None
        # Attributes
        self.theta = None
        self.n_iter = None
        self.n_samples = None
        self.n_features = None
        self.cost = None
        self.coef_ = None
        self.intercept_ = None
        
    def _coordinate_descent(self, X, Y):
        np.random.seed(self.random_state)
        convergence = False
        cost = [np.inf]*2
        
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
            theta = np.random.randn(self.n_features + 1) # random initialization for theta values, from Gaussian distribution
            theta[0] = 0
        else:
            X_b = X
            theta = np.random.randn(self.n_features)
        
        for iteration in range(int(self.max_iter)):
            for i in range(self.n_features+1):
                if self.fit_intercept and i == 0: # Lasso does not regularize the intercept
                    theta[0] = np.sum(Y - np.dot(X_b[:, 1:], theta[1:]))/self.n_samples
                else:
                    Xi = X_b[:, i]
                    theta[i] = 0
                    rho = np.dot(Xi.T, Y - np.dot(X_b, theta))
                    theta[i] = soft_threshold(rho, self.alpha*self.n_samples)/np.square(X_b[:, i]).sum()            

            # convergence criteria of 'cost function'
            cost.pop(0)
            cost.append(mse(X_b, theta, Y))            
            print(cost[1], end="\r")
            # print(theta)
            if np.abs(cost[0] - cost[1]) < self.tol:
                convergence = True
                break
            
        if convergence:
            print("Achieve Convergence!")
        else:
            print("Reached Max Iteration! Not Achieve Convergence.")
            
        self.cost = cost[1]
        self.n_iter = iteration
        return theta
    
    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.theta = self._coordinate_descent(X, Y)
        return self
    
    def predict(self, X):
        if self.fit_intercept:
            X_b = np.c_[np.ones((len(X),1)), X]
        else:
            X_b = X
        return np.dot(X_b, self.theta)