"""Diagnostic BF16 invstd buffer and a fixed externally driven affine kernel."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from benchmarks.artgor_input_trace import _ordered_sum, MEAN_ORDERS


def chunked_pair_host(states,operation,*,devices=8,chunk_rows=256):
    if devices<=0 or chunk_rows<=0 or not len(states) or len(states)%(devices*chunk_rows):
        raise ValueError('invalid device/chunk partition')
    local=len(states)//devices
    partition=np.asarray(states).reshape(devices,local,*states.shape[1:])
    outputs=None
    for start in range(0,local,chunk_rows):
        chunk=partition[:,start:start+chunk_rows].reshape(devices*chunk_rows,*states.shape[1:])
        pair=tuple(np.asarray(x) for x in operation(chunk))
        if len(pair)!=2 or any(len(x)!=devices*chunk_rows for x in pair):
            raise ValueError('expected pair with unchanged row count')
        if outputs is None:
            outputs=tuple(np.empty((devices,local,*x.shape[1:]),dtype=x.dtype) for x in pair)
        for out,x in zip(outputs,pair):
            out[:,start:start+chunk_rows]=x.reshape(devices,chunk_rows,*x.shape[1:])
    return tuple(out.reshape(len(states),*out.shape[2:]) for out in outputs)


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


def _variance_pair_kernel(dense,mean,var_out,inv_out,*,epsilon,order):
    centered=dense[...].astype(jnp.float32)-mean[...].astype(jnp.float32)
    variance=_ordered_sum(centered*centered,order)/dense.shape[1]
    invstd=jax.lax.rsqrt(variance+jnp.asarray(epsilon,jnp.bfloat16).astype(jnp.float32))
    var_out[...]=jnp.broadcast_to(variance,dense.shape)
    inv_out[...]=jnp.broadcast_to(invstd.astype(jnp.bfloat16),dense.shape)


def variance_pair(dense,mean,*,epsilon=1e-5,order='native',bm=128,interpret=False):
    if order not in MEAN_ORDERS:
        raise ValueError('unknown reduction order')
    matrix=pl.BlockSpec((bm,dense.shape[1]),lambda i:(i.astype(jnp.int32),jnp.int32(0)))
    return pl.pallas_call(functools.partial(_variance_pair_kernel,epsilon=epsilon,order=order),
        out_shape=(jax.ShapeDtypeStruct(dense.shape,jnp.float32),jax.ShapeDtypeStruct(dense.shape,jnp.bfloat16)),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,in_specs=[matrix,matrix],
            out_specs=(matrix,matrix),grid=(dense.shape[0]//bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret,name=f'variance_pair_{order}',
    )(dense,mean)


def _variance_rsqrt_kernel(variance,out,*,epsilon):
    out[...]=jax.lax.rsqrt(variance[...].astype(jnp.float32)
        +jnp.asarray(epsilon,jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)


def variance_rsqrt(variance,*,epsilon=1e-5,bm=128,interpret=False):
    matrix=pl.BlockSpec((bm,variance.shape[1]),lambda i:(i.astype(jnp.int32),jnp.int32(0)))
    return pl.pallas_call(functools.partial(_variance_rsqrt_kernel,epsilon=epsilon),
        out_shape=jax.ShapeDtypeStruct(variance.shape,jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0,in_specs=[matrix],
            out_specs=matrix,grid=(variance.shape[0]//bm,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=interpret,name='variance_rsqrt_replay',
    )(variance)
