"""Diagnostic BF16 invstd buffer and a fixed externally driven affine kernel."""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _invstd_kernel(dense, mean, out, *, epsilon):
    centered = dense[...].astype(jnp.float32) - mean[...].astype(jnp.float32)
    variance = jnp.sum(centered * centered, axis=1, keepdims=True) / dense.shape[1]
    invstd = jax.lax.rsqrt(variance + jnp.asarray(epsilon,jnp.bfloat16).astype(jnp.float32))
    out[...] = jnp.broadcast_to(invstd.astype(jnp.bfloat16),dense.shape)


def invstd_buffer(dense, mean, *, epsilon=1e-5, bm=128, interpret=False):
    matrix = pl.BlockSpec((bm,dense.shape[1]),lambda i:(i.astype(jnp.int32),jnp.int32(0)))
    return pl.pallas_call(
        functools.partial(_invstd_kernel,epsilon=epsilon),
        out_shape=jax.ShapeDtypeStruct(dense.shape,jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[matrix,matrix],out_specs=matrix,grid=(dense.shape[0]//bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret,name='captured_invstd_bf16',
    )(dense,mean)


def _affine_kernel(dense,mean,invstd,scale,bias,out):
    centered=dense[...].astype(jnp.float32)-mean[...].astype(jnp.float32)
    out[...] = jnp.maximum(centered*invstd[...].astype(jnp.float32)
        *scale[...].astype(jnp.float32)[None,:]+bias[...].astype(jnp.float32)[None,:],0).astype(jnp.bfloat16)


def external_invstd_affine(dense,mean,invstd,scale,bias,*,bm=128,interpret=False):
    matrix=pl.BlockSpec((bm,dense.shape[1]),lambda i:(i.astype(jnp.int32),jnp.int32(0)))
    vector=pl.BlockSpec((dense.shape[1],),lambda i:(jnp.int32(0),))
    return pl.pallas_call(_affine_kernel,
        out_shape=jax.ShapeDtypeStruct(dense.shape,jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,
            in_specs=[matrix,matrix,matrix,vector,vector],out_specs=matrix,grid=(dense.shape[0]//bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret,name='external_invstd_affine',
    )(dense,mean,invstd,scale,bias)
