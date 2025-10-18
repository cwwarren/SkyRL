import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any


@jax.tree_util.register_dataclass
@dataclass
class AdamState:
    m: Any  # pytree of (N, ...) matching p
    v: Any  # pytree of (N, ...) matching p
    b1: jax.Array  # (N,)
    b2: jax.Array  # (N,)


@jax.tree_util.register_dataclass
@dataclass
class AdamParams:
    lr: jax.Array  # (N,)
    b1: jax.Array  # (N,)
    b2: jax.Array  # (N,)
    eps: jax.Array  # (N,)
    wd: jax.Array  # (N,)


def adamw_init(N, p, lr, b1, b2, eps, wd, dtype=jnp.float32):
    state = AdamState(
        m=jax.tree.map(lambda w: jnp.zeros_like(w, dtype=dtype), p),
        v=jax.tree.map(lambda w: jnp.zeros_like(w, dtype=dtype), p),
        b1=jnp.ones((N,), dtype=dtype),
        b2=jnp.ones((N,), dtype=dtype),
    )
    hyper = AdamParams(
        lr=jnp.full((N,), lr, dtype=dtype),
        b1=jnp.full((N,), b1, dtype=dtype),
        b2=jnp.full((N,), b2, dtype=dtype),
        eps=jnp.full((N,), eps, dtype=dtype),
        wd=jnp.full((N,), wd, dtype=dtype),
    )
    return state, hyper


def _x(a: jax.Array, r: jax.Array) -> jax.Array:
    return a.reshape(a.shape + (1,) * (r.ndim - 1))


def _adamw_leaf(p, g, m, v, b1, b2, h, mask):
    mask_ = _x(mask, m)
    b1_, b2_, eps, lr, wd = _x(h.b1, m), _x(h.b2, v), _x(h.eps, p), _x(h.lr, p), _x(h.wd, p)
    b1, b2 = _x(b1, m), _x(b2, v)

    m_ = b1_ * m + (1.0 - b1_) * g.astype(m.dtype)
    v_ = b2_ * v + (1.0 - b2_) * g.astype(m.dtype) ** 2

    m_hat = m_ / jnp.where(mask_, 1.0 - b1, 1.0)
    v_hat = v_ / jnp.where(mask_, 1.0 - b2, 1.0)
    step = lr * (m_hat / (jnp.sqrt(v_hat) + eps))
    p_ = p.astype(m.dtype) * (1.0 - lr * wd) - step
    return (
        jnp.where(mask_, p_.astype(p.dtype), p),
        jnp.where(mask_, m_, m),
        jnp.where(mask_, v_, v),
    )


def adamw_step(p, g, s: AdamState, h: AdamParams, mask: jax.Array):
    mask = mask.astype(jnp.bool_)
    b1, b2 = jnp.where(mask, s.b1 * h.b1, s.b1), jnp.where(mask, s.b2 * h.b2, s.b2)
    tree = jax.tree.map(lambda p, g, m, v: _adamw_leaf(p, g, m, v, b1, b2, h, mask), p, g, s.m, s.v)
    p, m, v = jax.tree.transpose(jax.tree.structure(p), jax.tree.structure((0, 0, 0)), tree)
    return p, AdamState(m, v, b1, b2)
