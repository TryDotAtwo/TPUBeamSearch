"""Validate response gate evidence by reconstructing original host fixtures."""
from benchmarks.beam_response_epoch_fixture import fixtures,expected_epoch
from benchmarks.beam_external_dedup_probe import digest


def validate_report(report,source_sha):
    errors=[]
    if report.get('source_sha')!=source_sha or report.get('exact') is not True:
        errors.append('source or completion')
    devices=report.get('devices',[])
    if (len(devices)!=8 or any(type(d.get('id')) is not int for d in devices)
            or len({d.get('id') for d in devices})!=8
            or any(not str(d.get('kind','')).startswith('TPU') for d in devices)):
        errors.append('eight distinct TPU devices')
    runtime=report.get('runtime',{})
    if any(not isinstance(runtime.get(k),str) or not runtime[k] for k in ('jax','jaxlib','libtpu')):
        errors.append('runtime')
    cases=report.get('cases',[])
    expected_cases=list(fixtures())
    if [r.get('name') for r in cases]!=[name for name,_ in expected_cases]:
        return errors+['case sequence']
    for row,(name,inputs) in zip(cases,expected_cases):
        if row.get('exact') is not True or row.get('input_sha256')!=digest(inputs):
            errors.append(name+': input or completion')
        epochs=row.get('epochs',[])
        if [e.get('epoch') for e in epochs]!=[0,1,2]:
            errors.append(name+': epochs')
            continue
        wire,requests,control,validation=inputs
        for index,item in enumerate(epochs):
            expected=expected_epoch(wire,requests,control[:,0,0],
                (control[:,1,0]!=0)|(validation[:,0,0]!=0),index)
            mismatch=item.get('mismatches',[])
            valid=(len(mismatch)==2 and all(isinstance(r,list) and len(r)==8
                and all(type(v) is int and v==0 for v in r) for r in mismatch))
            if (not valid or item.get('exact') is not True
                    or item.get('expected_sha256')!=digest(expected)
                    or item.get('output_sha256')!=digest(expected)):
                errors.append(f'{name}: epoch{index}')
    return errors
