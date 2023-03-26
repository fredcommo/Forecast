import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np
from dateutil.relativedelta import relativedelta

from forecast.utils import(
    date_to_timestamp,
    timestamp_to_daily_sin_cos,
    timestamp_to_weekly_sin_cos,
    timestamp_to_monthly_sin_cos,
    timestamp_to_yearly_sin_cos
)

from forecast.optimize import(
    optimize,
    get_best_model,
    train_best_model,
    predict,
    predict_future,
    wape,
    plot
)

class TimeSeries():
    def __init__(self, df, y, date_col="Date Time",
                 datetime_transform= ["daily", "yearly"],
                 lags=16, test_size=24,
                 future_period=12, future_freq="hours"):
        
        self.date_time = df[date_col] if is_datetime(df[date_col]) else pd.to_datetime(df[date_col])
        self.y = df[y]
        self.X = df.drop(columns = [y, date_col])
        self.datetime_transform = datetime_transform
        self.lags = lags
        self.test_size = test_size
        self.future_period = future_period
        self.future_freq = future_freq
        
        self._add_lags()
        self._create_Xfuture()
        self._datetime_transform()
        self._train_test_split()
    

    def _add_lags(self):

        if self.lags < 1:
            return

        def _lag_to_frame(y, i):
            return pd.DataFrame({f"lag_{i}": [np.nan]*i + list(y[:-i])})

        Lags = [_lag_to_frame(self.y, i) for i in range(1, self.lags + 1, 1)]
        Lags = pd.concat(Lags, axis=1)
        self.X = pd.concat([self.X, Lags], axis=1)

        idx = ~self.X[f"lag_{self.lags}"].isna()
        self.date_time = self.date_time[idx]
        self.X = self.X[idx].reset_index(drop=True)
        self.y = self.y[idx].reset_index(drop=True)


    def _create_Xfuture(self):

        def _futures(y, i):
            return pd.DataFrame({f"lag_{i}": list(y)[-i:]})

        def _relativedelta(freq, i):
            if freq == "hours":
                return relativedelta(hours=i)
            elif freq == "days":
                return relativedelta(days=i)
            elif freq == "weeks":
                return relativedelta(weeks=i)
            elif freq == "months":
                return relativedelta(months=i)
            elif freq == "years":
                return relativedelta(years=i)

        last_date = self.date_time.max()
        freq = self.future_freq
        y = self.y[-self.lags:]
        future_dates = [last_date + _relativedelta(freq, i) \
                            for i in range(1, self.future_period + 1, 1)]
        future_dates = pd.to_datetime(future_dates)
        
        self.date_time = np.concatenate((self.date_time, future_dates))
        self.date_time = pd.to_datetime(self.date_time)

        futures = [_futures(y, i) for i in range(1, self.lags + 1, 1)]
        futures = pd.concat(futures, axis=1)[:self.future_period]
        if self.future_period > self.lags:
            n = self.future_period - self.lags
            empty = np.empty((n, futures.shape[1]))
            empty[:] = np.nan
            empty = pd.DataFrame(empty, columns=futures.columns)
            futures = pd.concat([futures, empty], axis=0)

        self.X = pd.concat([self.X, futures]).reset_index(drop=True)

        
    def _datetime_transform(self):
        timestamp_s = date_to_timestamp(self.date_time)
        
        if "daily" in self.datetime_transform:
            self.X['Dayly_sin'], self.X['Dayly_cos'] = timestamp_to_daily_sin_cos(timestamp_s)
        if "weekly" in self.datetime_transform:
            self.X['Weekly_sin'], self.X['Weekly_cos'] = timestamp_to_weekly_sin_cos(timestamp_s)
        if "monthly" in self.datetime_transform:
            self.X['Monthly_sin'], self.X['Monthly_cos'] = timestamp_to_monthly_sin_cos(timestamp_s)
        if "yearly" in self.datetime_transform:
            self.X['Yearly_sin'], self.X['Yearly_cos'] = timestamp_to_yearly_sin_cos(timestamp_s)
        

    def _train_test_split(self):
        self.Xfuture = self.X.iloc[-self.future_period:, : ]
        self.X = self.X.iloc[:-self.future_period, : ]
        self.Xtrain = self.X.iloc[:-self.test_size, : ]
        self.ytrain = self.y[:-self.test_size]
        self.Xtest = self.X.iloc[-self.test_size:, : ]
        self.ytest = self.y[-self.test_size:]
        

    def get_Xtrain(self):
        return np.array(self.Xtrain)

    def get_Xtest(self):
        return np.array(self.Xtest)

    def get_Xfuture(self):
        return np.array(self.Xfuture)

    def get_ytrain(self):
        return np.array(self.ytrain).reshape(-1, 1)
    
    def get_ytest(self):
        return np.array(self.ytest).reshape(-1, 1)

#########################################
# Additional methods for class timeSeries
#########################################
TimeSeries.optimize = optimize
TimeSeries.get_best_model = get_best_model
TimeSeries.train_best_model = train_best_model
TimeSeries.predict = predict
TimeSeries.predict_future = predict_future
TimeSeries.wape = wape
TimeSeries.plot = plot
