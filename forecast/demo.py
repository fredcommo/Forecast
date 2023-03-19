import os
import pandas as pd

from forecast.utils import (
    date_to_timestamp,
    timestamp_to_daily_sin_cos,
    timestamp_to_yearly_sin_cos,
    compute_daily_avg,
    compute_weekly_avg,
    compute_monthly_avg,
    compute_quarterly_avg
    )

from forecast.timeseries import TimeSeries

def main():
    path = "forecast/data"
    filename = 'jena_climate_2009_2016_simpl.csv'
    df = pd.read_csv(os.path.join(path, filename), parse_dates=["Date Time"])

    #####################################################
    # Compute daily, weekly, monthly, quarterly temp avg.
    #####################################################
    date_col = "Date Time"
    y = "T (degC)"
    df = compute_daily_avg(df, date_col=date_col, y=y)
    df = compute_weekly_avg(df, date_col=date_col, y=y)
    # df = compute_monthly_avg(df, date_col=date_col, y=y)
    # df = compute_quarterly_avg(df, date_col=date_col, y=y)

    #####################################################
    # Turn dates into timestamps, then into sin/cos
    #####################################################
    timestamp_s = date_to_timestamp(df, "Date Time")
    df['Day sin'], df['Day cos'] = timestamp_to_daily_sin_cos(timestamp_s)
    df['Year sin'], df['Year cos'] = timestamp_to_yearly_sin_cos(timestamp_s)

    print(f"n rows: {df.shape[0]}")
    print(f"n features: {df.shape[1] - 1}\n")
    print(df.head())


    #####################################################
    # Create a TimeSeries object
    #####################################################
    ts = TimeSeries(df, y='T (degC)', lags=7*24)

    #####################################################
    # Optimize, then compute and plot predictions on test set
    #####################################################
    # model_list=["XGBReg"]
    model_list=["Ridge", "Lasso", "ElasticNet"]
    # model_list=["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
    # model_list=["LinearRegression", "Ridge", "Lasso", "ElasticNet", "XGBReg"]

    print("Testing:\n-", "\n- ".join(model_list))
    ts.optimize(model_list=model_list, timeout=20*60, n_trials=50)
    ts.train_best_model()
    ts.plot()


if __name__ == "__main__":
    main()
