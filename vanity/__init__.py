__version__ = '0.1.0'
__all__ = ["normalize_vanity"]

from anndata import AnnData
from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp

from .model import fit


def normalize_vanity(adata: AnnData, inplace: bool=True):
    """
    Estimate log-expression values using a Bayesian regression model.

    Args:
      adata: A `anndata.AnnData` object with raw counts in the `X` matrix.
      inplace: Flag controlling whether to modify `adata` in place or return estimates.

    Returns:
      If `inplace` is True, modifies `adata` by replacing `adata.X` with
      posterior mean log-expression values. In addition, adds the following entries
      to `adata.layers`.
        - "vanity_var": posterior variance estimates for the log-expression values
        - "vanity_log_fc_mean": posterior mean estimates of log fold-change versus the mean
        - "vanity_log_fc_var": posterior variance estimates of log fold-change

    If `inplace` is False, instead return a dict with these entries.
    """

    # TODO: check that this is count data

    log_expr_mean, log_expr_var, log_fc_mean, log_fc_var = fit(adata.X)

    if inplace:
        adata.X = log_expr_mean
        adata.layers["vanity_var"] = log_expr_var
        adata.layers["vanity_log_fc_mean"] = log_fc_mean
        adata.layers["vanity_log_fc_var"] = log_fc_var
    else:
        return {
            "X": log_expr_mean,
            "vanity_var": log_expr_var,
            "vanity_log_fc_mean": log_fc_mean,
            "vanity_log_fc_var": log_fc_var,
        }
