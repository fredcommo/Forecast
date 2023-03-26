import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np

def date_to_timestamp(dates):
    """
    Transform dates to datetime format if not already, then to timestamps in sec
    """
    if not is_datetime(dates):
        dates = pd.to_datetime(dates)
    return dates.map(pd.Timestamp.timestamp)

###########################################
# A series of utils to turn timestamps into sin/cos signals
###########################################

def timestamp_to_daily_sin_cos(timestamp_s):
    day = 24*60*60
    return np.sin(timestamp_s * (2 * np.pi / day)), \
            np.cos(timestamp_s * (2 * np.pi / day))

def timestamp_to_weekly_sin_cos(timestamp_s):
    day = 24*60*60
    week = day*7
    return np.sin(timestamp_s * (2 * np.pi / week)), \
            np.cos(timestamp_s * (2 * np.pi / week))

def timestamp_to_monthly_sin_cos(timestamp_s):
    day = 24*60*60
    month = day*30
    return np.sin(timestamp_s * (2 * np.pi / month)), \
            np.cos(timestamp_s * (2 * np.pi / month))

def timestamp_to_yearly_sin_cos(timestamp_s):
    day = 24*60*60
    year = (365.2425)*day
    return np.sin(timestamp_s * (2 * np.pi / year)), \
            np.cos(timestamp_s * (2 * np.pi / year))

def compute_daily_avg(df, date_col, y):
    df_c = df.copy()
    df_c["Day"] = df_c[date_col].dt.floor("D")

    daily_avg = df_c \
        .groupby(pd.Grouper(key=date_col, freq="D", closed='left', label='left'))[y] \
        .mean() \
        .reset_index(drop=False)
    daily_avg.rename(columns={date_col: "Day", y: "Daily_avg"}, inplace=True)

    df_c = df_c.merge(daily_avg, on=["Day"], how="left")
    df_c.drop(columns=["Day"], inplace=True)

    return df_c

def compute_weekly_avg(df, date_col, y):
    df_c = df.copy()
    df_c["Week"] = df_c[date_col].dt.to_period("W").dt.start_time

    weekly_avg = df_c \
        .groupby(pd.Grouper(key=date_col, freq="W-MON", closed='left', label='left'))[y] \
        .mean() \
        .reset_index(drop=False)
    weekly_avg[date_col] = weekly_avg[date_col].dt.to_period("W").dt.start_time
    weekly_avg.rename(columns={date_col: "Week", y: "Weekly_avg"}, inplace=True)

    df_c = df_c.merge(weekly_avg, on=["Week"], how="left")
    df_c.drop(columns=["Week"], inplace=True)

    return df_c

def compute_monthly_avg(df, date_col, y):
    df_c = df.copy()
    df_c["Month"] = df_c[date_col].dt.to_period("M").dt.start_time

    monthly_avg = df_c \
        .groupby(pd.Grouper(key=date_col, freq="MS", closed='left', label='left'))[y] \
        .mean() \
        .reset_index(drop=False)
    monthly_avg.rename(columns={date_col: "Month", y: "Monthly_avg"}, inplace=True)

    df_c = df_c.merge(monthly_avg, on=["Month"], how="left")
    df_c.drop(columns=["Month"], inplace=True)

    return df_c

def compute_quarterly_avg(df, date_col, y):
    df_c = df.copy()
    df_c["Quarter"] = df_c[date_col].dt.to_period("Q").dt.start_time

    quaterly_avg = df_c \
        .groupby(pd.Grouper(key=date_col, freq="QS", closed='left', label='left'))[y] \
        .mean() \
        .reset_index(drop=False)

    quaterly_avg.rename(columns={date_col: "Quarter", y: "Quarterly_avg"}, inplace=True)

    df_c = df_c.merge(quaterly_avg, on=["Quarter"], how="left")
    df_c.drop(columns=["Quarter"], inplace=True)

    return df_c
