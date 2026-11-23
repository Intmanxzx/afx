"""Bounded local data artifacts. No pickle, executable graph or remote loading."""
import hashlib
import json
import os
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024


def write_arrays(path, samples):
    import numpy as np
    import torch
    arrays, logical_dtypes = {}, {}
    for i, tensors in enumerate(samples):
        for j, t in enumerate(tensors):
            key = f's{i}_{j}'
            t = t.detach().cpu().resolve_conj().contiguous()
            if t.dtype == torch.bfloat16:
                logical_dtypes[key] = 'bfloat16'
                t = t.view(torch.int16)
            arrays[key] = t.numpy()
    if sum(a.nbytes for a in arrays.values()) > MAX_BYTES:
        raise ValueError('output buffer budget exceeded')
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez(stream, **arrays)
    if tmp.stat().st_size > MAX_BYTES:
        tmp.unlink()
        raise ValueError('artifact budget exceeded')
    os.replace(tmp, path)
    return {'version': 1, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'counts': [len(s) for s in samples], 'bytes': path.stat().st_size,
            'logical_dtypes': logical_dtypes}


def read_arrays(directory, ref):
    import numpy as np
    import torch
    if ref['version'] != 1 or Path(ref['file']).name != ref['file']:
        raise ValueError('invalid local artifact reference')
    path = Path(directory)/ref['file']
    if path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError('unsafe artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('artifact digest mismatch')
    # Check uncompressed members before NumPy allocates any arrays.
    import zipfile
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError('unpacked artifact budget exceeded')
    with np.load(path, allow_pickle=False) as data:
        if sum(data[k].nbytes for k in data.files) > MAX_BYTES:
            raise ValueError('unpacked artifact budget exceeded')
        def tensor(key):
            value = torch.from_numpy(data[key].copy())
            logical = ref.get('logical_dtypes', {}).get(key)
            if logical is not None:
                if logical != 'bfloat16' or value.dtype != torch.int16:
                    raise ValueError('invalid logical storage dtype')
                value = value.view(torch.bfloat16)
            return value
        return [tuple(tensor(f's{i}_{j}') for j in range(n))
                for i, n in enumerate(ref['counts'])]


def write_bundle(directory, bundle):
    from autofusex.workloads.fuzz.case_spec import encode
    from autofusex.workloads.fuzz.generation import workload_id
    raw=b''.join(bundle.storages)
    if len(raw)>MAX_BYTES:raise ValueError('Case storage budget exceeded')
    path=Path(directory)/'inputs.bin'
    path.write_bytes(raw)
    return {'spec':encode(bundle.spec),'workload_id':workload_id(bundle),
            'storage_file':path.name,'storage_sizes':[len(b) for b in bundle.storages],
            'storage_sha256':hashlib.sha256(raw).hexdigest()}


def bundle_from(value, directory):
    from autofusex.workloads.fuzz.case_spec import decode
    from autofusex.workloads.fuzz.contracts import CaseBundle
    from autofusex.workloads.fuzz.generation import workload_id
    if Path(value['storage_file']).name!=value['storage_file']:
        raise ValueError('invalid Case artifact reference')
    path=Path(directory)/value['storage_file']
    if path.is_symlink() or path.stat().st_size>MAX_BYTES:raise ValueError('unsafe Case artifact')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=value['storage_sha256']:raise ValueError('Case storage digest mismatch')
    lengths=value['storage_sizes']
    if len(lengths)>10000 or any(type(n) is not int or n<0 for n in lengths) or sum(lengths)!=len(raw):raise ValueError('invalid storage manifest')
    storages=[];offset=0
    for n in lengths:
        storages.append(raw[offset:offset+n]);offset+=n
    bundle=CaseBundle(decode(value['spec']),tuple(storages))
    if workload_id(bundle)!=value['workload_id']:raise ValueError('shared Case digest mismatch')
    return bundle
