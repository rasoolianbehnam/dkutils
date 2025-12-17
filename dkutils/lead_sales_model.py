import dkutils
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

XCOLS = [
    "consumerconfidenceindex",
    "primeinterestrate",
    "unemploymentrate",
    "automobileinventorytosalesratio",
    "consumerpriceindex",
    "consumerpriceindexgasoline",
    "grossdomesticproduct",
    "holiday",
]
SALES_LOGIT_INTERCEPT = -2.7


def series(**kwargs):
    return pd.Series(dict(**kwargs))


class ClosingRateModel(dkutils.model.ProphetModel):
    def sample_components(
        self,
    ):
        seasonality = yield from self.sample_seasonality(hierarchical=False)
        exogenous = yield from self.sample_exogenous()
        trend = yield from self.sample_trend()

        return self.get_logits(seasonality, exogenous, trend)

    def get_logits(self, seasonality, exogenous, trend, **kwargs):
        ratio = seasonality + trend + exogenous
        ratio = jnp.einsum("...n->n...", ratio)[self.data.train.train.values]
        ratio = jnp.einsum("n...->...n", ratio) + SALES_LOGIT_INTERCEPT
        return ratio

    def model(self, total_leads_log):
        ratio = yield from self.sample_components()
        total_leads = jnp.exp(
            total_leads_log * self.data.stats.loc["std", "total_leads_log"]
            + self.data.stats.loc["mean", "total_leads_log"]
        )
        yield tfd.Independent(
            tfd.Binomial(total_count=total_leads, logits=ratio),
            reinterpreted_batch_ndims=1,
            name="total_sold_obs",
        )


# Extract training data for leads and sales from the prepared dataset
def extract_training_data(data):
    """Extract and transform training data"""
    total_leads_log = data.data.loc[data.train.train, "total_leads_log"]
    total_leads = np.exp(
        total_leads_log * data.stats.loc["std", "total_leads_log"]
        + data.stats.loc["mean", "total_leads_log"]
    )
    total_sold = data.data.loc[data.train.train, "total_sold"]
    data["target"] = series(
        total_leads_log=total_leads_log, total_leads=total_leads, total_sold=total_sold
    )


# Define individual models for leads and sales
def create_joint_model(data):
    """Create joint model for leads and sales prediction"""
    model_leads = dkutils.model.ProphetModel(
        data, target_column="total_leads_log", xcols=XCOLS
    )
    model_sales = ClosingRateModel(data, target_column="total_sold", xcols=XCOLS)

    # Define the joint model for leads and sales
    @tfd.JointDistributionCoroutine
    def model():
        total_leads_log = yield from model_leads.model()
        yield from model_sales.model(total_leads_log=total_leads_log)

    return series(
        leads=model_leads,
        sales=model_sales,
        joint_pinned=model.experimental_pin(
            total_leads_log_obs=data.target.total_leads_log.values,
            total_sold_obs=data.target.total_sold.values,
        ),
    )


def compute_logits_sales(data, models):
    d = {k.replace("total_sold_", ""): v for k, v in data.posterior.dict.items()}
    logits = models.sales.get_logits(
        models.sales.get_seasonality(**d),
        models.sales.get_exogenous(**d),
        models.sales.get_trend(**d),
    )
    data["predictions"] = series(sale_logits=logits)


def compute_predictions(data):
    data["train_orig"] = data["train"].copy()
    data["train"] = data.train.assign(train=True)
    extract_training_data(data)
    models = create_joint_model(data)
    res = models.joint_pinned.unpin("total_leads_log_obs", "total_sold_obs").sample(
        **data.posterior.dict, seed=jax.random.PRNGKey(0)
    )
    compute_logits_sales(data, models)
    data["train"] = data["train_orig"]
    del data["train_orig"]
    data["predictions"]["total_leads_log"] = res.total_leads_log_obs
    data["predictions"]["total_leads"] = np.exp(
        res.total_leads_log_obs * data.stats.loc["std", "total_leads_log"]
        + data.stats.loc["mean", "total_leads_log"]
    )
    data["predictions"]["total_sold"] = res.total_sold_obs


def get_components(data, target):
    models = create_joint_model(data)
    repl = "total_leads_log_" if target == "leads" else "total_sold_"
    a = (
        models[target]
        .get_exogenous_components(
            **{k.replace(repl, ""): v for k, v in data.posterior.dict.items()}
        )
        .mean((0, 1))
    )
    b = (
        models[target]
        .get_seasonality(
            **{k.replace(repl, ""): v for k, v in data.posterior.dict.items()}
        )
        .mean((0, 1))[..., None]
    )
    c = (
        models[target]
        .get_trend(**{k.replace(repl, ""): v for k, v in data.posterior.dict.items()})
        .mean((0, 1))[..., None]
    )
    return pd.DataFrame(
        jnp.concat([a, b, c], axis=1),
        columns=[*XCOLS, "seasonality", "trend"],
        index=data.data.index,
    )


def compute_components(data):
    leads_components = get_components(data, "leads")
    sales_components = get_components(data, "sales")
    data["components"] = {}
    data["components"]["total_leads_log"] = (
        leads_components * data.stats.loc["std", "total_leads_log"]
    ).assign(intercept=data.stats.loc["mean", "total_leads_log"])
    data["components"]["sale_logits"] = sales_components.assign(
        intercept=SALES_LOGIT_INTERCEPT
    )


class Foo:
    def __init__(self, data):
        self.data = data
        d = data.posterior.dict
        models = create_joint_model(data)
        tmp = models["leads"](False)
        self.d = {k: d[k] for k in tmp.event_shape._asdict().keys() if k in d}

    def __call__(self, X):
        idx = [pd.Timestamp(_) for _ in X[:, 0].astype("int")]
        X = pd.DataFrame(X[:, 1:], columns=XCOLS, index=idx)
        idx = X.index
        df = (X - self.data.stats.loc["mean", XCOLS]) / self.data.stats.loc[
            "std", XCOLS
        ]
        new_train = self.data.train.copy()
        new_train["train"] = True
        new_data = pd.Series(dict(data=df, train=new_train))
        m = dkutils.model.ProphetModel(
            new_data, target_column="total_leads_log", xcols=XCOLS
        )
        out = m(False).sample(**self.d, seed=jax.random.PRNGKey(13)).total_leads_log_obs
        # return pd.Series(out.mean((0, 1)), index=sorted_idx).loc[idx]

        out = jnp.exp(
            out * self.data.stats.loc["std", "total_leads_log"]
            + self.data.stats.loc["mean", "total_leads_log"]
        ).mean((0, 1))
        out = pd.Series(out, index=idx)
        return out
