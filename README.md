
# Vanity

This is a straightforward "normalization" method for scRNA-Seq and similar data,
based on [Sanity](https://github.com/jmbreda/Sanity)

    Breda, Jérémie, Mihaela Zavolan, and Erik van Nimwegen. "Bayesian inference of gene expression states from single-cell RNA-seq data." Nature Biotechnology 39.8 (2021): 1008-1016.

Vanity implements this model in jax using standard mean-field variational
inference to dramatically speed up inference, particularly when running on a
GPU, and it works nicely with the anndata/scanpy ecosystem. Overall the goal is
a faster more convenient Sanity for my own use, and hopefully of some interest
to others.

Vanity seems to largely agree with with Sanity, with a slight tendency to
produce lower values. This hasn't been exhaustively tested yet, so proceed with
some caution.

![plot comparing estimates produced by sanity and vanity](https://github.com/dcjones/vanity/blob/main/agreement.png?raw=true)

Basic usage:
```python
from vanity import normalize_vanity

normalize_vanity(adata) # where adata is an anndata.AnnData object

```