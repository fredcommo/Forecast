import numpy as np

def wape(y, yhat):
    if any(y < 0):
        m = np.nanmin(y)
        y = y - m + 1
        yhat = yhat - m + 1
        
    abs_diff = np.abs(y - yhat)
    return (1 - np.nansum(abs_diff) / np.nansum(y))