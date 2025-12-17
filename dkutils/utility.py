import joblib
import dataiku
import os
import tempfile

import pandas as pd
from io import StringIO
from pathlib import Path


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


def download_file(path, directory):
    folder, file = path.split("/")
    directory = Path(directory)
    with open(directory / file, "wb") as f:
        f.write(dataiku.Folder(folder).get_download_stream(file).read())


def read_obj(path, read_fn=joblib.load):
    folder, file = path.split("/")
    with tempfile.NamedTemporaryFile() as f:
        f.write(dataiku.Folder(folder).get_download_stream(file).read())
        out = read_fn(f.name)
    return out


def upload_file(file, path):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    dataiku.Folder(dirname).upload_file(basename, file)


def upload_obj(obj, path, save_fn=joblib.dump):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        save_fn(obj, temp_dir / basename)
        dataiku.Folder(dirname).upload_file(basename, temp_dir / basename)
