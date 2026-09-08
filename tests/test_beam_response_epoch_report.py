import copy
import pytest


def valid_report():
    from benchmarks.beam_response_epoch_fixture import fixtures,expected_epoch
    from benchmarks.beam_external_dedup_probe import digest
    report=dict(source_sha='a'*40,exact=True,runtime=dict(jax='x',jaxlib='x',libtpu='x'),
        devices=[dict(id=i,kind='TPU v5 lite') for i in range(8)],cases=[])
    for name,inputs in fixtures():
        wire,requests,control,validation=inputs
        row=dict(name=name,input_sha256=digest(inputs),exact=True,epochs=[])
        for epoch in range(3):
            expected=expected_epoch(wire,requests,control[:,0,0],
                (control[:,1,0]!=0)|(validation[:,0,0]!=0),epoch)
            row['epochs'].append(dict(epoch=epoch,exact=True,mismatches=[[0]*8,[0]*8],
                expected_sha256=digest(expected),output_sha256=digest(expected)))
        report['cases'].append(row)
    return report


@pytest.mark.parametrize('mutation',['none','missing','hash','rank','device','sha'])
def test_report_rejects_missing_or_inconsistent_evidence(mutation):
    from benchmarks.beam_response_epoch_report import validate_report
    report=copy.deepcopy(valid_report())
    if mutation=='missing': report['cases'][-1]['epochs'].pop()
    if mutation=='hash': report['cases'][0]['epochs'][0]['output_sha256']='bad'
    if mutation=='rank': report['cases'][0]['epochs'][0]['mismatches'][0][7]=1
    if mutation=='device': report['devices'][7]['id']=0
    if mutation=='sha': report['source_sha']='b'*40
    assert bool(validate_report(report,'a'*40))==(mutation!='none')
