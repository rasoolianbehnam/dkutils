import joblib
import dataiku
import os

import pandas as pd
from io import StringIO


def read_data(data_name):
    return lower(dataiku.Dataset(data_name).get_dataframe())


def lower(df):
    out = df.copy()
    out.columns = [x.lower() for x in out.columns]
    return out


def read_remote_csv(path):
    folder_name, path = path.split("/")
    handler = dataiku.Folder(folder_name)
    with handler.get_download_stream(path) as f:
        data = f.read()
    market_inputs = str(data, "utf-8")
    market_inputs_imported = StringIO(market_inputs)
    return pd.read_csv(market_inputs_imported)


def read_remote_file(path, read_fn=joblib.load):
    folder, file = path.split("/")
    with open("tmp", "wb") as f:
        f.write(dataiku.Folder(folder).get_download_stream(file).read())
    return read_fn("tmp")


def upload_to(obj, path, save_fn=joblib.dump):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    save_fn(obj, basename)
    dataiku.Folder(dirname).upload_file(basename, basename)
    os.remove(basename)
