import os
import pkg_resources
import pandas as pd
import glob


def get_data(file, **kwargs):
    stream = pkg_resources.resource_stream(__name__, f"data/{file}.csv")
    return pd.read_csv(stream, **kwargs)

def files_list():
    path = str(__name__).split(".")[0]
    return [os.path.basename(f).replace(".csv", "") for f in glob.glob(f"{path}/data/*.csv")]
