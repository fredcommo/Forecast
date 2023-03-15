import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np

from optimize import(
    optimize,
    get_best_model,
    train_best_model,
    predict,
    plot
)

class TimeSeries():
    def __init__(self, df, y, date_col="Date Time", lags=16, test_len=24):
        self.date_time = df[date_col] if is_datetime(df[date_col]) else pd.to_datetime(df[date_col])
        self.y = df[y]
        self.X = df.drop(columns=[y, date_col])
        self.lags = lags
        self.test_len = test_len
        
        self._add_lags()
        self._train_test_split()
        
    def _add_lags(self):
        if self.lags < 1:
            return
        for i in range(1, self.lags + 1, 1):
            self.X[f"lag_{i}"] = [np.nan]*i + list(self.y[:-i])

        idx = ~self.X[f"lag_{self.lags}"].isna()
        self.date_time = self.date_time[idx]
        self.X = self.X[idx]
        self.y = self.y[idx]
        
    def _train_test_split(self):
        self.Xtrain = self.X.iloc[:-self.test_len,:]
        self.ytrain = self.y[:-self.test_len]
        self.Xtest = self.X.iloc[-self.test_len:,:]
        self.ytest = self.y[-self.test_len:]

    def get_Xtrain(self):
        return self.Xtrain

    def get_Xtest(self):
        return self.Xtest

    def get_ytrain(self):
        return self.ytrain
    
    def get_ytest(self):
        return self.ytest
    
TimeSeries.optimize = optimize
TimeSeries.get_best_model = get_best_model
TimeSeries.train_best_model = train_best_model
TimeSeries.predict = predict
TimeSeries.plot = plot
