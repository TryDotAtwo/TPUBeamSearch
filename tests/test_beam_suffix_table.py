import importlib.util
import numpy as np
import pytest


def test_suffix_table_preserves_bfs_chain_ids_and_low_first_move_encoding():
    assert importlib.util.find_spec('tpu_beam_search.beam_suffix_table') is not None
    from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table
    table = prepare_k2_suffix_table(move_count=2,radius=2)
    assert table.count == 7
    assert table.words.shape == (3,128)
    np.testing.assert_array_equal(table.words[0,:7],[0,0,1,0,32,1,33])
    np.testing.assert_array_equal(table.words[2,:7],[0,1,1,2,2,2,2])
    assert not table.words[1].any() and not table.words[:,7:].any()
    assert not table.words.flags.writeable


@pytest.mark.parametrize('moves,count',[(24,14425),(30,27931)])
def test_suffix_table_actual_move_counts_and_padding(moves,count):
    from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table
    table = prepare_k2_suffix_table(move_count=moves,radius=3,max_count=count)
    assert table.count == count and table.words.shape[1]%128 == 0
    assert table.words[0,count-1] == (moves-1)*(1+32+1024)
    assert table.words[2,count-1] == 3
    assert not table.words[:,count:].any()
    with pytest.raises(ValueError):
        prepare_k2_suffix_table(move_count=moves,radius=3,max_count=count-1)


@pytest.mark.parametrize('kwargs',[{'move_count':33,'radius':1},
    {'move_count':24,'radius':4},{'move_count':0,'radius':1},
    {'move_count':24,'radius':-1},{'move_count':True,'radius':1}])
def test_suffix_table_rejects_unrepresentable_or_unbounded_geometry(kwargs):
    from tpu_beam_search.beam_suffix_table import prepare_k2_suffix_table
    with pytest.raises(ValueError):
        prepare_k2_suffix_table(**kwargs)
