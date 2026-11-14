#!/usr/bin/env python3
"""Build the project's patched NNSmith wheel in a disposable publisher directory.

Never run this on a user's checkout or install into their Python environment.
The wheel is not published automatically. Distribution requires license review.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def tree_hash(files):
    digest=hashlib.sha256()
    for name,data in sorted(files):digest.update(name.encode()+b'\0'+data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,help='Existing pinned NNSmith checkout; copied, never modified')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    installer=(root/'autofusex/nnsmith_integration/install_af_fuzz.sh').read_text()
    commit=re.search(r'NNSMITH_COMMIT="([a-f0-9]+)"',installer).group(1)
    source=Path(args.source).resolve();output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise ValueError('NNSmith source is not the pinned commit')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('publisher requires a clean pinned checkout')
    order=re.search(r'PATCH_ORDER=\((.*?)\)',installer,re.S).group(1).split()
    expected=re.search(r"SOURCE_SHA256\s*=\s*'([a-f0-9]{64})'",(root/'autofusex/fuzz/generation/runtime.py').read_text()).group(1)
    with tempfile.TemporaryDirectory(prefix='nnsmith-wheel-') as temp:
        working=Path(temp)/'source'
        shutil.copytree(source,working,ignore=shutil.ignore_patterns('.git','__pycache__','.venv'))
        for name in order:
            patch=root/'autofusex/nnsmith_integration/patches'/(name+'.patch')
            subprocess.run(['patch','-p1','--forward','--no-backup-if-mismatch','-i',str(patch)],cwd=working,check=True)
        package=working/'nnsmith'
        if tree_hash((p.relative_to(package).as_posix(),p.read_bytes()) for p in package.rglob('*.py'))!=expected:
            raise ValueError('patched source fingerprint mismatch')
        # SCM-generated _version.py is not part of the frozen source tree.
        metadata=working/'pyproject.toml'
        metadata.write_text('\n'.join(line for line in metadata.read_text().splitlines()
                                     if not line.lstrip().startswith('write_to'))+'\n')
        dist=Path(temp)/'dist'
        subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(dist)],cwd=working,check=True,
                       env=dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION='0.1.0'))
        wheels=list(dist.glob('*.whl'))
        if len(wheels)!=1:raise ValueError('expected exactly one NNSmith wheel')
        with zipfile.ZipFile(wheels[0]) as archive:
            files=[(name[len('nnsmith/'):],archive.read(name)) for name in archive.namelist() if name.startswith('nnsmith/') and name.endswith('.py')]
            if tree_hash(files)!=expected:raise ValueError('wheel changed frozen source fingerprint')
        target=output/wheels[0].name
        if target.exists():raise ValueError('refusing to overwrite existing release artifact')
        shutil.copy2(wheels[0],target)
        import json
        print(json.dumps(dict(path=str(target),source_commit=commit,source_sha256=expected,wheel_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__=='__main__':main()
