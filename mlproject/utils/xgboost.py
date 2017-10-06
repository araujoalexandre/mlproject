"""
    custom XGBoost Objective Function
"""
from functools import partial
import numpy as np

__all__ = ['FairObj', 'HuberObj', 'GiniObj']

def FairObj(preds, dtrain, c=2):
    labels = dtrain.get_label()
    diff = preds - labels
    grad = c*diff / (np.abs(diff)+c)
    hess = c**2 / (np.abs(diff)+c)**2
    return grad, hess

def HuberObj(preds, dtrain, delta=2):

    def ApproxHessWithGaussian(pred, label, grad):
        diff = pred - label
        x = abs(diff)
        a = 2.0 * np.fabs(grad)
        c = max((abs(pred) + abs(label)) * 0.01, 1.0e-10)
        return np.exp((-x*x) / (2.0*c*c))*a / (c*np.sqrt(2*np.pi))

    def HuberComp(pred, label):
        diff = pred - label
        if (np.abs(diff) <= delta):
            grad = diff
            hess = 1
        else:
            if (diff >= 0.):
                grad = delta;
            else:
                grad = -delta
            hess = ApproxHessWithGaussian(pred, label, grad)
        return grad, hess
    
    labels = dtrain.get_label()
    grad, hess = [], []
    for y, t in zip(preds, labels):
        g, h = HuberComp(y, t)
        grad.append(g)
        hess.append(h)
    return grad, hess

def GiniObj(pred, y):

    def gini(y, pred):
        g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
        g = g[np.lexsort((g[:,2], -1*g[:,1]))]
        gs = g[:,0].cumsum().sum() / g[:,0].sum()
        gs -= (len(y) + 1) / 2.
        return gs / len(y)

    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)