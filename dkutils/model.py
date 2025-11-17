import jax
import pandas as pd
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
jnp = jax.numpy

root = tfd.JointDistributionCoroutine.Root

XCOLS = [
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


def get_week_of_month(s):
    out = (
        pd.DataFrame(
            dict(date=s.values, month=s.dt.month.values, year=s.dt.year.values)
        )
        .groupby(["year", "month"])
        .apply(
            lambda df: df.sort("date").assign(week_of_month=range(len(df)))[
                ["date", "week_of_month"]
            ],
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return out.set_index("date").loc[s, "week_of_month"]


def gen_fourier_basis(t, p=365.25, n=4):
    x = 2 * jnp.pi * (jnp.arange(n) + 1) * t[:, None] / p
    return jnp.concatenate((jnp.cos(x), jnp.sin(x)), axis=1)


def Ind(d, reinterpreted_batch_ndims=1, **kwargs):
    return tfd.Independent(
        d, reinterpreted_batch_ndims=reinterpreted_batch_ndims, **kwargs
    )


def event(self, n_dims=1, name=None):
    name = name or self.name
    return Ind(self, n_dims, name=name)


class ProphetModel:
    def __init__(self, data):
        self.data = data
        df = data.data
        trainInfo = data.train
        index = df.index.to_series()
        start_day = df.index.min()
        scale = 365.25 * 4
        t = (index - start_day).dt.days.values
        self.S = gen_fourier_basis(t, n=3).astype("float32")
        self.t = t / scale
        self.s = (
            trainInfo.changepoint[trainInfo.changepoint].index - start_day
        ).to_series().dt.days.values / scale
        self.A = (self.t[:, None] > self.s).astype("float32")

        self.X = jnp.array(df[XCOLS].values.astype("float32"))
        self.seasonality_switch = data.train.seasonality_switch.values
        self.n_seasonality_switches = self.seasonality_switch.max() + 1
        self.changepoints = trainInfo.changepoint.cumsum().values * 0
        self.n_changepoints = trainInfo.changepoint.sum() + 1

    def model(self):

        seasonality = yield from self.sample_seasonality(hierarchical=False)
        exogenous = yield from self.sample_exogenous()
        trend = yield from self.sample_trend()
        noise_sigma = yield root(tfd.HalfNormal(scale=1.0, name="noise_sigma"))

        y_hat = seasonality + trend + exogenous

        y_hat = jnp.einsum("...n->n...", y_hat)[self.data.train.train.values]
        y_hat = jnp.einsum("n...->...n", y_hat)
        yield tfd.Independent(
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="obs",
        )

    def get_exogenous(self, alpha, **kwargs):
        exogenous = jnp.einsum("ij,...j->...i", self.X, alpha)
        return exogenous

    def sample_exogenous(self):
        d = self.X.shape[-1]
        alpha = yield tfd.Normal(0, 0.5, name="alpha").Sample(d).root
        return self.get_exogenous(alpha)

    def get_seasonality(self, beta_0, beta_1, beta, **kwarg):
        beta = beta * beta_1[..., None] + beta_0[..., None]
        beta = beta[..., self.seasonality_switch]
        seasonality = jnp.einsum("nd,...dn->...n", self.S, beta)
        return seasonality

    def sample_seasonality(self, hierarchical=False):
        if hierarchical:
            beta_0 = yield root(
                tfd.Sample(
                    tfd.Normal(0.0, 0.1), sample_shape=self.S.shape[-1], name="beta_0"
                )
            )
            beta_1 = yield root(
                tfd.Sample(
                    tfd.HalfNormal(0.1), sample_shape=self.S.shape[-1], name="beta_1"
                )
            )
        else:
            beta_0 = jnp.array(0)
            beta_1 = jnp.array(0.1)
        beta = yield root(
            tfd.Sample(
                tfd.Normal(0, 1),
                sample_shape=[self.S.shape[-1], self.n_seasonality_switches],
                name="beta",
            )
        )
        return self.get_seasonality(beta_0, beta_1, beta)

    def get_trend(self, k, m, tau, delta, **kwargs):
        delta = delta * tau[..., None]
        growth_rate = k[..., None] + jnp.einsum("ij,...j->...i", self.A, delta)
        offset = m[..., None] + jnp.einsum("ij,...j->...i", self.A, -self.s * delta)
        trend = growth_rate * self.t + offset
        return trend

    def sample_trend(self):
        k = yield root(tfd.Normal(0, 1, name="k"))  # Slope at 0
        m = yield root(tfd.Normal(0, 1, name="m"))
        tau = yield root(tfd.Exponential(1, name="tau"))
        delta = yield root(
            tfd.Sample(
                tfd.Laplace(0.0, tau), sample_shape=self.A.shape[1], name="delta"
            )
        )
        return self.get_trend(k, m, tau, delta)

    def sample_intercept(self):
        intercept = (
            yield tfd.Normal(0, 1).Sample(self.n_changepoints, name="intercept").root
        )
        return intercept.cumsum(axis=-1)[..., self.changepoints]

    def sample_economy(self, hidden_dim=1):
        d = self.X.shape[-1]
        initial_state = (
            yield tfd.Normal(0, 1).Sample(hidden_dim, name="hist_initial_state").root
        )
        A = yield tfd.Normal(0, 1).Sample([d, hidden_dim], name="hist_A").root
        A2 = (
            yield tfd.Normal(0, 1).Sample([hidden_dim, hidden_dim], name="hist_A2").root
        )
        b = yield tfd.Normal(0, 1).Sample(hidden_dim, name="hist_b").root
        O = yield tfd.Normal(0, 1, name="hist_O").Sample(hidden_dim + 1).root
        return self.get_economy(initial_state, A, A2, b, O)

    def get_economy(self, initial_state, A, A2, b, O, **kwargs):

        def step(carry, _):
            prev_state, i = carry
            prev_state = jnp.einsum("...dD,...d->...D", A2, prev_state)
            out = jax.nn.sigmoid(
                prev_state + jnp.einsum("...dk,d->...k", A, self.X[i]) + b
            )
            return (out, i + 1), out

        states = jax.scipy.special.logit(
            jax.lax.scan(step, (initial_state, 0), length=len(self.X))[1]
        )
        states = jnp.einsum("t...k->...tk", states)
        return jnp.einsum("...k,...tk->...t", O[..., :-1], states) + O[..., -1:]

    def __call__(self, pinned=True):
        data = self.data
        model = tfd.JointDistributionCoroutine(self.model)
        if pinned:
            return model.experimental_pin(
                obs=data.data["visit"][data.train.train].values
            )
        return model
