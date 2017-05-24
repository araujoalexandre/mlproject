"""
__file__

    LogisticRegression.py

__description__

    Logistic Regression with newton-cg method
    
__author__

    Araujo Alexandre < aaraujo001@gmail.com >

"""


import numpy as np
from scipy import stats


class LogisticRegressionCustom:

    def __init__(self, fit_intercept=True, maxiter=100, tol=1e-4):

        self.fit_intercept=fit_intercept
        self.maxiter = maxiter
        self.tol = tol
        self.coefs_ = None
        self.n_iter_ = None


    def _sigmoid(self, x):
        """
            compute the sigmoid of x
        """
        return 1. / (1 + np.exp(-x))


    def _conjugate_gradient(self, grad, hess, maxiter=100, tol=1e-4):
        """
            Solve iteratively the linear system 'A . x = b'
            with a conjugate gradient descent.

            A = hess, b = - grad
            
            https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """

        xi = np.zeros(len(grad))
        ri = grad
        pi = ri

        dri0 = ri @ ri

        i = 0
        while i <= maxiter:

            Ap = (hess @ pi) + pi

            alphai = dri0 / (pi @ Ap)

            xi = xi + alphai * pi
            ri = ri - alphai * Ap

            if np.sum(np.abs(ri)) <= tol:
                break

            dri1 = ri @ ri
            betai = dri1 / dri0 
            pi = ri + betai * pi

            i += 1
            dri0 = dri1

        return xi


    def _newton_cg(self, X, y):
        """
            train model on X

            INPUT:
                X: ndarray, array
                y: ndarray, array
        """

        m, n = X.shape
        beta = np.zeros(n)

        if self.fit_intercept:
            ones = np.ones((m, 1))
            X = np.column_stack((ones, X))

        y[y == 0] = -1
        k = 0

        while k < self.maxiter:

            # calculate the gradient    
            yz = y * (X @ beta)
            z = self._sigmoid(yz)
            z0 = (z - 1) * y

            grad = (X.T @ z0) + beta
            hess = X.T @ ((z * (1 - z)).reshape(-1, 1) * X)

            absgrad = np.abs(grad)
            if np.max(absgrad) < self.tol:
                break

            xi = self._conjugate_gradient(grad, hess)
            beta = beta - xi

            k += 1

        return beta, k


    def fit(self, X, y, method='newton_cg'):
        """
            train model on X

            INPUT:
                X: ndarray, array
                y: ndarray, array
        """
        if method == 'newton_cg':
            self.coefs_, self.n_iter_ = self._newton_cg(X, y)


    def predict_proba(self, X):
        """
            compute proba of X base on self.coefs_
        """

        # if not isinstance(self.coefs_, list):
        #     raise("You need to call fit metod first")

        y_hat = self._sigmoid(X @ self.coefs_).flatten()
        return np.column_stack((1 - y_hat, y_hat))


def logloss(y_true, y_hat, eps=10e-15):
    """ FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    
    if not isinstance(y_hat, np.ndarray):
        y_hat = np.array(y_hat)

    y_hat = np.clip(y_hat, eps, 1. - eps)
    ll = sum(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))
    return - ll / len(y_hat)




def mixed_feature_selection(X_train, y_train, X_cv, y_cv, verbose=False):
    """
        XXX
    """

    m, n = X_train.shape

    # evaluation of each feature
    scores = []
    for i in range(n):

        model = LogisticRegressionCustom(fit_intercept=False)
        model.fit(X_train[:,i], y_train)

        ll = logloss(y_train, model.predict_proba(X_train[:,i]))
        scores.append(ll)

    feature_order = np.argsort(scores)[::-1]

    X_train_select = np.ones((m, 1))
    X_cv_select = np.ones((m, 1))

    greedy_subset = []
    best_logloss = np.finfo(np.float32).max

    while feature_order:

        keep = False

        index_feat = feature_order.pop()
        greedy_subset.append(index_feat)

        X_train_select = np.column_stack((X_train_select, X_train[:, index_feat]))
        X_cv_select = np.column_stack((X_cv_select, X_cv[:, index_feat]))
        
        model = LogisticRegressionCustom(fit_intercept=False)
        model.fit(X_train_select, y_train)

        # y_hat_train = model.predict_proba(X_train_select)
        # train_score = logloss(y_train, y_hat_train)

        y_hat_cv = model.predict_proba(X_cv_select)
        cv_score = logloss(y_cv, y_hat_cv)

        v = np.diagflat((1 - y_hat_cv) * y_hat_cv)
        se = np.sqrt(np.diag(np.linalg.inv(X_cv_select.T @ v @ X_cv_select)))

        t_stat = model.coef_ / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), m - n))

        # get p_value of last feature
        # if p_value > 0.1, remove feature
        # if p_value is small, check is score gets better
        if p_values[-1] > 0.1:
            keep = False
        elif cv_score < best_logloss:
            keep = True
            best_logloss = cv_score

        if not keep:
            X_train_select = X_train_select[:, :-1]
            X_cv_select = X_cv_select[:, :-1]
            greedy_subset.remove(index_feat)

    return greedy_subset

