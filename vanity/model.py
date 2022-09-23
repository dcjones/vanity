

from typing import Any
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
import sys

Array = Any


def fit(X: Array, quiet: bool=False):
    ncells, ngenes = X.shape

    Nc = np.squeeze(np.asarray(X.sum(axis=1))) # [ncells]

    chunk_size = min(ngenes, max(1, int(1e8 / 4 / ncells)))
    quiet or print(f"chunk size: {chunk_size}")

    δ_loc_chunks = []
    δ_scale_chunks = []

    for gene_from in range(0, ngenes, chunk_size):
        gene_to = min(gene_from + chunk_size, ngenes)

        Xchunk = X[:,gene_from:gene_to]
        Xchunk = jax.device_put(
            Xchunk if isinstance(Xchunk, np.ndarray) else Xchunk.toarray()).astype(jnp.float32)

        δ_loc, δ_scale = fit_chunk(Xchunk, Nc)

        δ_loc_chunks.append(δ_loc)
        δ_scale_chunks.append(δ_loc)

    δ_loc = np.concatenate(δ_loc_chunks, axis=1)
    δ_scale = np.concatenate(δ_scale_chunks, axis=1)

    assert δ_loc.shape == X.shape
    assert δ_scale.shape == X.shape

    return δ_loc, δ_scale


# X: [ncell, ngene]
# Nc: [ncell]
def fit_chunk(X: Array, Nc: Array, maxiter: int=4000, seed: int=9876543210):
    key = jax.random.PRNGKey(seed)
    key, ρ_key, h_key = jax.random.split(key, 3)

    ncells, ngenes = X.shape

    print((ncells, ngenes))

    params = {
        "δ_loc": jnp.zeros((ncells, ngenes)),
        "δ_scale": jnp.full((ncells, ngenes), -2.0),
        "v_loc": jnp.zeros(ngenes),
        "v_scale": jnp.full(ngenes, -4.0)
    }

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    Ng = jnp.sum(X, axis=0) # [ngenes]

    # precompute the log-likelihood terms that are constant wrt δ
    c = \
        jax.scipy.special.gammaln(Ng + 1) + \
        jnp.einsum("cg,c->g", X, jnp.log(Nc)) + \
        jnp.sum(jax.scipy.special.gammaln(X + 1), axis=0) # [ngenes]

    def logprob(q, X, Nc, Ng, c):
        δ = q["δ"]
        v = q["v"]

        prior_lp = \
            jnp.sum(tfd.Normal(loc=0.0, scale=v).log_prob(δ), axis=(1,2)) + \
            jnp.sum(tfd.HalfCauchy(loc=0.0, scale=0.1).log_prob(v), axis=1)

        # sanity's log likelihood derivation
        likelihood_lp = jnp.sum(
            jnp.expand_dims(c, 0) + \
            jnp.einsum("cg,bcg->bg", X, δ) - \
            (Ng + 1) * jnp.log(jnp.einsum("c,bcg->bg", Nc, jnp.exp(δ))),
            axis=1)

        return prior_lp + likelihood_lp

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

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, ELBO: {loss_value}")

    # Not estimate log-expression and it's variance

    return (np.array(params["δ_loc"]), np.array(nn.softplus(params["δ_scale"])))


def surrogate_posterior(params):
    return tfd.JointDistributionNamed({
        "δ": tfd.Independent(tfd.Normal(
                loc=params["δ_loc"],
                scale=nn.softplus(params["δ_scale"])),
            reinterpreted_batch_ndims=2),
        "v": tfd.Independent(tfd.LogNormal(
                loc=params["v_loc"],
                scale=nn.softplus(params["v_scale"])),
            reinterpreted_batch_ndims=1)
    })
