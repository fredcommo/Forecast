import numpy as np

def wape(y, yhat):
    if any(y < 0):
        m = np.nanmin(y)
        y = y - m + 1
        yhat = yhat - m + 1
        
    abs_diff = np.abs(y - yhat)
    return (1 - np.nansum(abs_diff) / np.nansum(y))

def mape(y, yhat):
    if any(y < 0):
        m = np.nanmin(y)
        y = y - m + 1
        yhat = yhat - m + 1
        
    abs_diff = np.abs(y - yhat)
    return (1 - np.nansum(abs_diff / y))

def wmape(y, yhat):
    if any(y < 0):
        m = np.nanmin(y)
        y = y - m + 1
        yhat = yhat - m + 1

    weights = np.array([y_i/np.nansum(y) for y_i in y])
    return (1 - np.nansum(weights * np.abs(y - yhat) / y))


def score(self, what="test", method="wape"):
    y = self.get_ytest()
    yhat = self.predict(what)
    return eval(method)(y, yhat)
