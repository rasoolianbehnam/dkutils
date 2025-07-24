import os
import joblib
from pathlib import Path
import dkutils
import pandas as pd

from brutils.utility import *
from brutils.probabilistic.tfp_utils_jax import *

data_dictionoary = {
    "visits": '"{ca_snowflake_managed_db}"."{ca_snowflake_managed_schema}"."dss_{projectKey}_VISITS"',
    "MOODYS_MACRO_ECON_DATA": '"{ca_snowflake_managed_db}".{ca_snowflake_managed_schema}."dss_{projectKey}_MOODYS_MACRO_ECON_DATA"',
    "monthly_predictions": '"{ca_snowflake_managed_db}"."{ca_snowflake_managed_schema}"."dss_{projectKey}_MONTHLY_PREDICTIONS"',
}

xCols = [
    "predictions_nc",
    "predictions_uc",
    "consumerconfidenceindex",
    "primeinterestrate",
    "unemploymentrate",
    "automobileinventorytosalesratio",
    "consumerpriceindex",
    "consumerpriceindexgasoline",
    "grossdomesticproduct",
    "brand_media_spend",
    "holiday",
]


def lower(df):
    out = df.copy()
    out.columns = [x.lower() for x in out.columns]
    return out


def add_dir(path):
    pass


def read_data(data_name):
    data_path = data_dictionoary[data_name].format(**dkutils.variables)
    return dkutils.S.table(data_path).pandas()


def get_root():
    return Path("data/dataiku/{projectKey}".format(**dkutils.variables))


def read_remote_csv(path):
    return pd.read_csv(get_root() / path)


def read_remote_file(path):
    return joblib.load(get_root() / path)


def upload_to(obj, target_path):
    target_path = get_root() / target_path
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    if str(target_path).endswith(".csv") and hasattr(obj, "to_csv"):
        obj.to_csv(target_path)
    else:
        joblib.dump(obj, target_path)


def process(df, features, training_end):
    data = features[xCols].merge(df, left_index=True, right_index=True, how="left")
    df0 = data.resample("W").sum().iloc[1:][:"2027-03"]
    df0.loc[df0.index[df0.visit > 0].max(), ["visit"]] = 0

    trainInfo = df0.assign(train=False)[["train"]]
    trainInfo["changepoint"] = False
    # trainInfo.loc['2021':'2024-12', 'train'] = True
    trainInfo.loc["2021":training_end, "train"] = True

    # trainInfo.loc['2021-10-03', 'changepoint'] = True
    trainInfo.loc["2023-01-01", "changepoint"] = True
    trainInfo.loc["2024-06-02", "changepoint"] = True
    trainInfo.loc["2025-03-02", "changepoint"] = True
    # trainInfo.loc['2025-01-05', 'changepoint'] = True

    df = df0.copy()
    df["visit"] = np.where(df.visit > 0, df.visit, np.nan)
    stats = df[trainInfo.train].describe()
    df = (df - df[trainInfo.train].mean()) / df[trainInfo.train].std()

    return pd.Series(dict(data=df, original_data=df0, train=trainInfo, stats=stats))
