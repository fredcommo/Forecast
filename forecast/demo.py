import os
import pandas as pd

from utils import (
    date_to_timestamp,
    timestamp_to_daily_sin_cos,
    timestamp_to_yearly_sin_cos
    )

from timeseries import TimeSeries

def main():
    path = "data"
    filename = 'jena_climate_2009_2016_simpl.csv'
    df = pd.read_csv(os.path.join(path, filename), parse_dates=["Date Time"])

    # turn dates into timestamps, then into sin/cos
    timestamp_s = date_to_timestamp(df, "Date Time")
    df['Day sin'], df['Day cos'] = timestamp_to_daily_sin_cos(timestamp_s)
    df['Year sin'], df['Year cos'] = timestamp_to_yearly_sin_cos(timestamp_s)

    # Create a TimeSeries object
    ts = TimeSeries(df, y='T (degC)', lags=72)

    # Optimize, then compute and plot predictions on test set
    model_list=["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
    # model_list=["XGBReg"]
    ts.optimize(model_list=model_list, timeout=60, n_trials=10)
    ts.train_best_model()
    ts.plot()


if __name__ == "__main__":
    main()
