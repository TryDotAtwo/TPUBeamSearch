"""Static host geometry for one uint32 HBM arena; no allocation or alias proof."""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ScratchPlan:
    # All regions are (byte offset, padded byte size).
    common: tuple[int,int]
    select_temp: tuple[int,int]
    materialize_temp: tuple[int,int]
    stream_temp: tuple[int,int]
    stream_persistent: tuple[int,int]
    final_budget_bytes: int
    phase_bytes: tuple[int,int,int]

    @property
    def pool_bytes(self):
        return max(self.phase_bytes)

    @property
    def pool_shape(self):
        return (self.pool_bytes // 512,128)


def plan_scratch(*, common_bytes, select_temp_bytes, materialize_temp_bytes,
                 stream_temp_bytes, stream_persistent_bytes):
    """Plan three exclusive layouts with a live selection/materialization prefix.

    Sizes are caller-computed storage requirements, not logical record counts.
    A512-byte granule gives128 uint32 lanes; it is a chosen arena ABI, not a
    universal TPU allocation requirement. Frontier/weights/solved/stop are
    outside this pool. Phase changes require separately established drains.
    This planner does not prove compiler buffer donation or physical aliasing.
    """
    def padded(n):
        if isinstance(n,bool) or not isinstance(n,Integral) or not 0 <= n < 2**63-512:
            raise ValueError('scratch size must be a nonnegative bounded integer')
        return ((int(n)+511)//512)*512
    common,selection,material,streams,persistent = map(padded,
        (common_bytes,select_temp_bytes,materialize_temp_bytes,stream_temp_bytes,stream_persistent_bytes))
    final_budget = common+max(selection,material)
    persistent_offset = max(final_budget,streams)
    phases = (persistent_offset+persistent,common+selection,common+material)
    if max(phases) >= 2**63:
        raise ValueError('scratch total exceeds signed64 capacity')
    return ScratchPlan((0,common),(common,selection),(common,material),
        (0,streams),(persistent_offset,persistent),final_budget,phases)


def pallas_write_scratch_region(arena, values, *, region, interpret=False):
    """Replace one aligned region; preserve every other arena word.

    Explicit HBM input/output alias inside Pallas. Caller must donate the outer
    JIT argument for physical reuse, and establish exclusive lifetime before
    writing. This serialized helper is not a lifetime manager or overlap proof.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    offset,size = region
    if (any(isinstance(x,bool) or not isinstance(x,Integral) or x < 0 or x%512 for x in region)
            or arena.ndim != 2 or arena.shape[1] != 128 or arena.dtype != jnp.uint32
            or values.shape != (size//512,128) or values.dtype != jnp.uint32
            or offset+size > arena.size*4):
        raise ValueError('invalid scratch region or uint32 arena geometry')
    if size == 0:
        return arena
    start = int(offset)//512
    def write(old,data,out,stage,sem):
        row = pl.program_id(0)
        load = pltpu.make_async_copy(data.at[pl.ds(row,1),:],stage,sem)
        load.start()
        load.wait()
        store = pltpu.make_async_copy(stage,out.at[pl.ds(start+row,1),:],sem)
        store.start()
        store.wait()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(write,out_shape=jax.ShapeDtypeStruct(arena.shape,jnp.uint32),
        in_specs=(hbm,hbm),out_specs=hbm,input_output_aliases={0:0},grid=(int(size)//512,),
        scratch_shapes=(pltpu.VMEM((1,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_scratch_region_write')(arena,values)


def pallas_read_scratch_region(arena, *, region, interpret=False):
    """Copy an aligned arena region to an HBM output, not a zero-copy view.

    Stages must eventually consume arena refs directly to avoid this copy.
    This diagnostic adapter establishes content/offset correctness only.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    offset,size = region
    if (any(isinstance(x,bool) or not isinstance(x,Integral) or x < 0 or x%512 for x in region)
            or arena.ndim != 2 or arena.shape[1] != 128 or arena.dtype != jnp.uint32
            or offset+size > arena.size*4):
        raise ValueError('invalid scratch region or uint32 arena geometry')
    rows = int(size)//512
    if rows == 0:
        return jnp.empty((0,128),jnp.uint32)
    start = int(offset)//512
    def read(data,out,stage,sem):
        row = pl.program_id(0)
        load = pltpu.make_async_copy(data.at[pl.ds(start+row,1),:],stage,sem)
        load.start()
        load.wait()
        store = pltpu.make_async_copy(stage,out.at[pl.ds(row,1),:],sem)
        store.start()
        store.wait()
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(read,out_shape=jax.ShapeDtypeStruct((rows,128),jnp.uint32),
        in_specs=(hbm,),out_specs=hbm,grid=(rows,),
        scratch_shapes=(pltpu.VMEM((1,128),jnp.uint32),pltpu.SemaphoreType.DMA),
        interpret=interpret,name='beam_scratch_region_read')(arena)
