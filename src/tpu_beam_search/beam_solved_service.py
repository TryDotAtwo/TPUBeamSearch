"""Serialized solved append and common stop/error; not a full depth drain."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .beam_final_error_summary import pallas_final_error_summary
from .beam_s5_request import make_s5_request_call
from .beam_solved_collect import pallas_collect_solved


def make_solved_batch_service(mesh, *, local_rank, stop_on_found, interpret=False):
    """Every rank participates, including empty, rejected and already-stop ranks.

    Consumes checked Stream2Batch identities, not a second parent-base input.
    Completed/inflight work may still append after a stop request, matching
    the source attempted-count semantics. Caller prevents new job admission.
    Storage overflow remains an observable sticky solved-overflow flag, not
    silently promoted to a different source policy. Counter wrap is rejected
    before append and becomes common fatal. Both agreements are unconditional.

    The underlying solved arena is still the small VMEM diagnostic collector;
    this does not establish scalable HBM residency or a scratch/DMA barrier.
    """
    if type(local_rank) is not int or not 0 <= local_rank <= 255:
        raise ValueError('local rank must fit solved owner byte')
    agree = make_s5_request_call(mesh, interpret=interpret)

    def call(arena, control, batch, depth, prior_error):
        n = batch.words.shape[1]
        if (batch.words.shape != (8,n) or batch.solution_hashes.shape != (4,n)
                or batch.suffix_ids.shape != (1,n) or batch.found.shape != (1,n)
                or batch.error.shape != (2,128) or control.shape != (4,128)
                or depth.shape != (1,) or prior_error.shape != (1,128)
                or any(x.dtype != jnp.uint32 for x in (batch.words,
                    batch.solution_hashes,batch.suffix_ids,batch.found,
                    batch.error,control,depth,prior_error))):
            raise ValueError('invalid solved service ABI')
        hits = pallas_final_error_summary(batch.found, interpret=interpret)
        def admission(c,h,e,p,out):
            bad = ((e[0,0] != 0) | (p[0,0] != 0)
                   | (h[0,0] > jnp.uint32(0xffffffff)-c[0,0]))
            out[...] = jnp.where(jnp.arange(128)[None] == 0,
                                 bad.astype(jnp.uint32),jnp.uint32(0))
        local_error = pl.pallas_call(admission,
            out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
            interpret=interpret,name='beam_solved_append_admission')(
                control,hits,batch.error,prior_error)

        def records(w,h,s,d,f,e,out,flags):
            out[:4,:] = h[...]
            out[4:6,:] = w[4:6,:]
            out[6,:] = jnp.zeros((128,),jnp.uint32)
            out[7,:] = (w[7,:] & jnp.uint32(255)) | jnp.uint32((local_rank<<16)|(local_rank<<8))
            out[8,:] = jnp.full((128,),d[0],jnp.uint32)
            out[9,:] = s[0,:]
            flags[...] = jnp.where(e[0,0] == 0,f[...],jnp.uint32(0))
        tile = lambda rows: pl.BlockSpec((rows,128),lambda i:(0,i))
        rec,flags = pl.pallas_call(records,
            out_shape=(jax.ShapeDtypeStruct((10,n),jnp.uint32),
                       jax.ShapeDtypeStruct((1,n),jnp.uint32)),
            in_specs=(tile(8),tile(4),tile(1),pl.BlockSpec((1,)),tile(1),pl.BlockSpec((1,128))),
            out_specs=(tile(10),tile(1)),grid=(n//128,),interpret=interpret,
            name='beam_solved_batch_records')(batch.words,batch.solution_hashes,
                batch.suffix_ids,depth,batch.found,local_error)
        arena,control = pallas_collect_solved(arena,control,rec,flags,
            stop_on_found=stop_on_found,interpret=interpret)
        common_error = agree(local_error)
        def stop_request(c,e,out):
            out[...] = jnp.where(jnp.arange(128)[None] == 0,
                ((c[3,0] != 0) | (e[0,0] != 0)).astype(jnp.uint32),jnp.uint32(0))
        request = pl.pallas_call(stop_request,
            out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
            interpret=interpret,name='beam_solved_stop_request')(control,common_error)
        common_stop = agree(request)
        return arena,control,common_error,common_stop
    return call
