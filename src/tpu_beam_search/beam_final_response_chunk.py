"""One coordinated response epoch; returns private bytes, never publishes."""
from .beam_final_chunk import pallas_pack_final_chunk
from .beam_final_exchange import make_final_chunk_exchange
from .beam_final_receive import pallas_compact_final_received
from .beam_final_transport import pallas_planes_to_wire


def make_final_response_chunk_call(mesh,*,wire_width,interpret=False):
    """All ranks must call the same epochs, including empty/error ranks.

    Input is prepared grouped response payload plus three routing planes.
    Transport agrees on upstream/packing errors before payload transfer and
    drains its DMA before receive compaction. Returned bytes remain private:
    destination capacity, coverage, history and publication gates are separate.
    This serialized composition does not claim DMA/compute overlap.
    """
    if type(wire_width) is not int or wire_width<=0 or wire_width%128:
        raise ValueError('expected aligned response wire width')
    planes=wire_width//4
    exchange=make_final_chunk_exchange(mesh,planes=planes,interpret=interpret)
    def call(grouped,intervals,error,chunk):
        if grouped.ndim!=2 or grouped.shape[0]!=planes+3:
            raise ValueError('invalid grouped response planes')
        payload,controls=pallas_pack_final_chunk(grouped[:planes],intervals,chunk,
            world_size=mesh.size,prior_error=error,interpret=interpret)
        snapshots,counts,common_error=exchange(payload,controls)
        received,status=pallas_compact_final_received(snapshots,counts,common_error,
            interpret=interpret)
        return pallas_planes_to_wire(received,interpret=interpret),status
    return call
