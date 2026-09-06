"""CPU interpreter composition; host snapshot assembly is NOT TPU exchange."""
import numpy as np
import jax.numpy as jnp


def test_eight_logical_rank_selection_matches_phase_rank_shard_order():
    from tpu_beam_search.beam_final_phase import pallas_final_phase_masks
    from tpu_beam_search.beam_final_scan import pallas_final_phase_scan
    from tpu_beam_search.beam_final_prefix import pallas_final_prefixes
    from tpu_beam_search.beam_final_cap import pallas_final_cap
    from tpu_beam_search.beam_final_indices import pallas_final_indices
    counts = np.zeros((2,128),np.uint32)
    ordinals,reference = [],[[],[]]
    for rank in range(8):
        scores = np.zeros((2,128),np.uint32) # dirty padding would match less
        clean = np.zeros((1,128),np.uint32)
        clean[0,:2] = [3 if rank != 2 else 0,2 if rank != 2 else 0]
        for shard in range(2):
            for slot in range(int(clean[0,shard])):
                score = (rank+shard+slot)%3
                scores[shard,slot] = score
                if score<2:
                    reference[score].append((rank,shard,slot))
        masks = pallas_final_phase_masks(jnp.asarray(scores),jnp.asarray(clean),jnp.array([1],jnp.uint32),interpret=True)
        ordinal,count = pallas_final_phase_scan(masks,interpret=True)
        ordinals.append(ordinal)
        counts[:,rank] = np.asarray(count)[:,0]
    bases,totals = pallas_final_prefixes(jnp.asarray(counts),world_size=8,interpret=True)
    beam = np.zeros((2,128),np.uint32)
    beam[0,0] = len(reference[0])+3
    keep,error = pallas_final_cap(totals,jnp.asarray(beam),interpret=True)
    assert int(error[0,0]) == 0
    selected = []
    for rank,ordinal in enumerate(ordinals):
        indices,valid = pallas_final_indices(ordinal,bases,keep,error,rank=rank,interpret=True)
        indices,valid = np.asarray(indices),np.asarray(valid)
        for phase,shard,slot in zip(*np.nonzero(valid)):
            index = int(indices[phase,0,shard,slot])+(int(indices[phase,1,shard,slot])<<32)
            selected.append((index,(rank,int(shard),int(slot))))
    selected.sort()
    assert [i for i,_ in selected] == list(range(int(beam[0,0])))
    assert [identity for _,identity in selected] == (reference[0]+reference[1])[:int(beam[0,0])]
