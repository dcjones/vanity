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
    Normalize using a Bayesian regression model.
    """

    # TODO: check that this is count data

    δ_loc, δ_scale = fit(adata.X)

    # TODO: we should do like sanity does and try to recover some
    # log expression values, rather than just fold changes. Though this is
    # totally suitable for running maxspin

    if inplace:
        adata.X = δ_loc
        adata.layers["vanity_var"] = δ_scale
    else:
        return { "X": δ_loc, "vanity_var": δ_scale }
