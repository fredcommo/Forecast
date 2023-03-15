import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import numpy as np

def date_to_timestamp(df, date_col = "Date Time"):
    """
    Transform dates to datetime, then timestamp in sec
    """
    date_time = df[date_col]
    if not is_datetime(date_time):
        date_time = pd.to_datetime(df[date_col])
    return date_time.map(pd.Timestamp.timestamp)

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
