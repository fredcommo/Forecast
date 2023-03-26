import os
import pandas as pd

from forecast.utils import (
    date_to_timestamp,
    timestamp_to_daily_sin_cos,
    timestamp_to_monthly_sin_cos,
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
    # df = compute_daily_avg(df, date_col=date_col, y=y)
    # df = compute_weekly_avg(df, date_col=date_col, y=y)
    # df = compute_monthly_avg(df, date_col=date_col, y=y)
    # df = compute_quarterly_avg(df, date_col=date_col, y=y)

    #####################################################
    # Turn dates into timestamps, then into sin/cos
    #####################################################
    timestamp_s = date_to_timestamp(df, "Date Time")
    df['Day sin'], df['Day cos'] = timestamp_to_daily_sin_cos(timestamp_s)
    # df['Month sin'], df['Month cos'] = timestamp_to_monthly_sin_cos(timestamp_s)
    df['Year sin'], df['Year cos'] = timestamp_to_yearly_sin_cos(timestamp_s)

    #####################################################
    # Define the list of models to optimize
    #####################################################
    # Current available models: "LinearRegression", "XGBReg", "Ridge", "ElasticNet", "XGBReg"
    model_list=["Lasso"]
    print("Testing:\n-", "\n- ".join(model_list))

    #####################################################
    # Create a TimeSeries object
    #####################################################
    ts = TimeSeries(df, y='T (degC)', lags=7*24)
    print(f"n rows: {ts.get_Xtrain().shape[0]}")
    print(f"n features: {ts.get_Xtrain().shape[1] - 1}\n")

    #####################################################
    # Optimize, then compute and plot predictions on test set
    #####################################################

    ts.optimize(model_list=model_list, timeout=2*60*60, n_trials=100)
    ts.train_best_model()
    ts.plot()


if __name__ == "__main__":
    main()

# Best XGBReg (quite slow)
# {
# "lags": 5*24,
#  "scoring": "neg_mean_absolute_percentage_error",
#  'learning_rate': 0.9,
#  'n_estimators': 500,
#  'reg_alpha': 0.3,
#  'reg_lambda': 0.1
# }

# Best Ridge
# {
# "lags": 5*24,
# 'alpha': 2.0,
# 'positive': False
# }

# Best Lasso
# {
# "lags": 5*24,
# 'alpha': 2.0,
# 'positive': False
# }
