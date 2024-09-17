"""Utility functions, including all functions related to
loss computation, optimization, sampling and inverse problems.
"""
import jax.numpy as jnp
from jax import vmap


def trunc_svd(X, rank, hermitian=False):
    (n_row, n_col) = jnp.shape(X)
    U, S, Vt = jnp.linalg.svd(X, hermitian=hermitian)
    return U[:, :rank], S[:rank], Vt[:rank, :]


def batch_matmul(A, B):
    return vmap(lambda A, B: A @ B)(A, B)


def batch_trace(A):
    return vmap(jnp.trace)(A)
