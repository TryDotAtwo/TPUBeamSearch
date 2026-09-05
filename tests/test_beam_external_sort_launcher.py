import ast
import json
from pathlib import Path
import re


def test_external_sort_launcher_is_private_eight_tpu_and_pins_source():
    folder = Path(__file__).resolve().parents[1] / 'kaggle_beam_external_sort'
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'trydotatwo/tpu-beam-external-sort-probe'
    assert metadata['is_private'] is True
    assert metadata['enable_tpu'] is True
    assert metadata['enable_gpu'] is False
    source = (folder / metadata['code_file']).read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert re.fullmatch(r'[0-9a-f]{40}', assignments['COMMIT_SHA'])
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, str)}
    assert {'jax[tpu]==0.10.2', 'jaxlib==0.10.2', 'libtpu==0.0.42.1',
            'benchmarks.beam_external_sort_probe', '--detach',
            '/kaggle/working/beam_external_sort'} <= strings
