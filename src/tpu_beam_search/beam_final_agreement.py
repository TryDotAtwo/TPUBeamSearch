"""Collective coverage decision, not a complete frontier publication barrier."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .beam_final_target_coverage import pallas_final_target_coverage
from .beam_final_error_summary import pallas_final_error_summary
from .beam_s5_request import make_s5_request_call


def make_final_coverage_agreement(mesh, *, interpret=False):
    """Every rank participates after all local final targets are available.

    Returns common error and local summary. Caller must also gate transport/
    history errors and drain all DMA before publishing or reusing scratch.
    This function neither writes frontier nor signals that those drains occurred.
    """
    agree = make_s5_request_call(mesh,interpret=interpret)
    def flag(summary,out):
        out[...] = jnp.where(jnp.arange(128)[None,:] == 0,
                            (summary[0,0] != 0).astype(jnp.uint32),jnp.uint32(0))
    flag_call = pl.pallas_call(flag,out_shape=jax.ShapeDtypeStruct((1,128),jnp.uint32),
        interpret=interpret,name='beam_final_coverage_flag')
    def call(targets,valid,count):
        reasons = pallas_final_target_coverage(targets,valid,count,interpret=interpret)
        summary = pallas_final_error_summary(reasons,interpret=interpret)
        return agree(flag_call(summary)),summary
    return call
