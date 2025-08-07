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
        self.S = gen_fourier_basis(t, n=2).astype("float32")
        self.t = t / scale
        self.s = (
            trainInfo.changepoint[trainInfo.changepoint].index - start_day
        ).to_series().dt.days.values / scale
        self.A = (self.t[:, None] > self.s).astype("float32")

        self.X = df[XCOLS].values.astype("float32")
        self.seasonality_switch = data.train.seasonality_switch.values
        self.n_seasonality_switches = self.seasonality_switch.max() + 1
        self.changepoints = trainInfo.changepoint.cumsum().values * 0
        self.n_changepoints = trainInfo.changepoint.sum() + 1

    def model(self):

        seasonality = yield from self.sample_seasonality(hierarchical=False)
        exogenous = yield from self.sample_exogenous()
        economy = yield from self.sample_economy()
        trend = yield from self.sample_trend()
        # intercept = yield from self.sample_intercept()
        noise_sigma = yield root(tfd.HalfNormal(scale=1.0, name="noise_sigma"))
        nu = yield tfd.Exponential(0.2, name="nu").root

        y_hat = exogenous + economy + seasonality + trend

        y_hat = jnp.einsum("...n->n...", y_hat)[self.data.train.train.values]
        y_hat = jnp.einsum("n...->...n", y_hat)
        yield tfd.Independent(
            tfd.StudentT(nu[..., None] + 1, y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="obs",
        )

    def get_exogenous(self, alpha, **kwargs):
        exogenous = jnp.einsum("ij,...j->...i", self.X[:, [0, 1, 10]], alpha)
        return exogenous

    def sample_exogenous(self):
        d = 3  # self.X.shape[-1]
        alpha_corr = yield tfd.CholeskyLKJ(d, 0.2, name="alpha_corr").root
        alpha_sigma = yield tfd.HalfNormal(1).Sample(d, name="alpha_sigma").root
        alpha_cov_tril = alpha_corr * alpha_sigma[..., None]

        # alpha_cov_tril = yield tfd.WishartTriL(
        #     d + 4,
        #     jnp.diag(jnp.ones(d) / 5),
        #     input_output_cholesky=True,
        #     name="alpha_cov_tril",
        # ).root
        alpha = yield tfd.MultivariateNormalTriL(0, alpha_cov_tril, name="alpha")
        return self.get_exogenous(alpha)

    # def sample_exogenous(self, f=1):
    #     ex_b = yield tfd.Normal(0, 0.1, name="ex_b").Sample([self.X.shape[-1], f]).root
    #     ex_c = yield tfd.Normal(0, 1, name="ex_c").Sample(self.X.shape[-1]).root
    #     ex_eps = yield tfd.Normal(0, 1).Sample(self.X.shape[-1] + f, name="ex_eps").root
    #     return self.get_exogenous(ex_b, ex_c, ex_eps, f)

    # def get_exogenous(self, ex_b, ex_c, ex_eps, f=1, **kwargs):
    #     alpha = ex_eps[..., :-f] * ex_c + jnp.einsum(
    #         "...df,...f->...d", ex_b, ex_eps[..., -f:]
    #     )
    #     exogenous = jnp.einsum("ij,...j->...i", self.X, alpha)
    #     return exogenous

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
        tau = yield root(tfd.Exponential(80, name="tau"))
        delta = yield root(
            tfd.Sample(tfd.Laplace(0.0, 1), sample_shape=self.A.shape[1], name="delta")
        )
        return self.get_trend(k, m, tau, delta)

    def sample_intercept(self):
        intercept = (
            yield tfd.Normal(0, 1).Sample(self.n_changepoints, name="intercept").root
        )
        return intercept.cumsum(axis=-1)[..., self.changepoints]

    def sample_economy(self):
        X = jnp.array(self.X[:, 2:-1])
        initial_state = yield tfd.Normal(0, 1, name="hist_initial_state").root
        A = yield tfd.Normal(0, 1).Sample(8, name="hist_A").root
        b = yield tfd.Normal(0, 1).Sample(1, name="hist_b").root
        O = yield tfd.Normal(0, 1, name="hist_O").Sample(2).root

        def step(carry, _):
            prev_state, i = carry
            out = prev_state + jnp.einsum("...d,d->...", A, X[i])
            return (out, i + 1), out

        states = jax.nn.sigmoid(
            jax.lax.scan(step, (initial_state, 0), length=len(self.X))[1]
        )
        states = jnp.einsum("t...->...t", states)
        return O[..., :1] * states + O[..., 1:]

    def __call__(self, pinned=True):
        data = self.data
        model = tfd.JointDistributionCoroutine(self.model)
        if pinned:
            return model.experimental_pin(
                obs=data.data["visit"][data.train.train].values
            )
        return model
