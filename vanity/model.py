

from typing import Any
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

Array = Any


def fit(X: Array, quiet: bool=False):
    ncells, ngenes = X.shape

    Nc = np.squeeze(np.asarray(X.sum(axis=1))) # [ncells]

    chunk_size = min(ngenes, max(1, int(1e8 / 4 / ncells)))
    quiet or print(f"chunk size: {chunk_size}")

    log_expr_mean_chunks = []
    log_expr_var_chunks = []
    log_fc_mean_chunks = []
    log_fc_var_chunks = []

    for gene_from in range(0, ngenes, chunk_size):
        gene_to = min(gene_from + chunk_size, ngenes)

        quiet or print(f"Normalizing genes {gene_from} to {gene_to}")

        Xchunk = X[:,gene_from:gene_to]
        Xchunk = jax.device_put(
            Xchunk if isinstance(Xchunk, np.ndarray) else Xchunk.toarray()).astype(jnp.float32)

        log_expr_mean, log_expr_var, log_fc_mean, log_fc_var = \
            fit_chunk(Xchunk, Nc, quiet=quiet)

        log_expr_mean_chunks.append(log_expr_mean)
        log_expr_var_chunks.append(log_expr_var)
        log_fc_mean_chunks.append(log_fc_mean)
        log_fc_var_chunks.append(log_fc_var)

    log_expr_mean = np.concatenate(log_expr_mean_chunks, axis=1)
    log_expr_var = np.concatenate(log_expr_var_chunks, axis=1)
    log_fc_mean = np.concatenate(log_fc_mean_chunks, axis=1)
    log_fc_var = np.concatenate(log_fc_var_chunks, axis=1)

    assert log_expr_mean.shape == X.shape
    assert log_expr_var.shape == X.shape
    assert log_fc_mean.shape == X.shape
    assert log_fc_var.shape == X.shape

    return (log_expr_mean, log_expr_var, log_fc_mean, log_fc_var)


# X: [ncell, ngene]
# Nc: [ncell]
def fit_chunk(X: Array, Nc: Array, maxiter: int=10000, seed: int=9876543210, quiet: bool=False):
    key = jax.random.PRNGKey(seed)

    ncells, ngenes = X.shape

    params = {
        "log_α_loc":
            jnp.mean(jnp.log(1e-8 + X / jnp.expand_dims(Nc, 1)), axis=0),
        "log_α_scale": jnp.full(ngenes, -2.0),
        "δ_loc": jnp.zeros((ncells, ngenes)),
        "δ_scale": jnp.full((ncells, ngenes), -2.0),
        "v_loc": jnp.zeros(ngenes),
        "v_scale": jnp.full(ngenes, -4.0)
    }

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    Ng = jnp.sum(X, axis=0) # [ngenes]

    # terms in the likelihood that are constant wrt δ and α
    c_g = \
        jnp.einsum("cg,c->g", X, jnp.log(Nc)) + \
        jnp.sum(jax.scipy.special.gammaln(X + 1), axis=0) # [ngenes]

    c = jnp.sum(c_g)

    def logprob(q, X, Nc, Ng, c):
        log_α = q["log_α"] # [batch, ngenes]
        δ = q["δ"] # [batch, ncells, ngenes]
        v = q["v"] # [batch, ngenes]

        # Like in sanity, we are adopting an improper prior on α
        # TODO: We might consider some very mild prior on log_α to prevent
        # it from shrinking towards -infinity
        prior_lp = \
            jnp.sum(tfd.Normal(loc=0.0, scale=v).log_prob(δ), axis=(1,2)) + \
            jnp.sum(tfd.HalfCauchy(loc=0.0, scale=0.1).log_prob(v), axis=1)

        # sanity's log likelihood derivation
        likelihood_lp = \
            jnp.einsum("cg,bcg->b", X, δ) + \
            jnp.einsum("g,bg->b", Ng, log_α) - \
            jnp.einsum("c,bcg->b", Nc, jnp.exp(jnp.expand_dims(log_α, 1) + δ))

        return prior_lp + likelihood_lp + c

    @jax.jit
    def step(params, opt_state, key, X, Nc, Ng, c):
        def loss(params):
            return tfp.vi.monte_carlo_variational_loss(
                lambda **q: logprob(q, X, Nc, Ng, c),
                surrogate_posterior(params),
                seed=key)

        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for epoch in range(maxiter):
        step_key, key = jax.random.split(key)
        params, opt_state, loss_value = \
            step(params, opt_state, step_key, X, Nc, Ng, c)

        # TODO: check for convergence and quit early if possible

        if not quiet and epoch % 100 == 0:
            print(f"Epoch: {epoch}, ELBO: {loss_value}")

    # Not estimate log-expression and it's variance
    δ_loc = params["δ_loc"]
    δ_scale = nn.softplus(params["δ_scale"])

    log_α_loc = params["log_α_loc"]
    log_α_scale = nn.softplus(params["log_α_scale"])

    log_expr_mean = jnp.expand_dims(log_α_loc, 0) + δ_loc
    log_expr_var = jnp.square(jnp.expand_dims(log_α_scale, 0)) + jnp.square(δ_scale)

    log_fc_mean = δ_loc
    log_fc_var = jnp.square(δ_scale)

    return (
        np.array(log_expr_mean),
        np.array(log_expr_var),
        np.array(log_fc_mean),
        np.array(log_fc_var))


def surrogate_posterior(params):
    return tfd.JointDistributionNamed({
        "log_α": tfd.Independent(tfd.Normal(
                loc=params["log_α_loc"],
                scale=nn.softplus(params["log_α_scale"])),
            reinterpreted_batch_ndims=1),
        "δ": tfd.Independent(tfd.Normal(
                loc=params["δ_loc"],
                scale=nn.softplus(params["δ_scale"])),
            reinterpreted_batch_ndims=2),
        "v": tfd.Independent(tfd.LogNormal(
                loc=params["v_loc"],
                scale=nn.softplus(params["v_scale"])),
            reinterpreted_batch_ndims=1)
    })
