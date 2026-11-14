"""Bounded source evidence; no Git mutations or secret content in metadata."""
import hashlib
from pathlib import Path
import subprocess


def evidence(source, inputs):
    source=Path(source).resolve();digest=hashlib.sha256();count=0;total=0
    from .artifacts import confined
    for relative in sorted(inputs):
        root=confined(source/relative,[source])
        paths=sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in paths:
            if path.is_symlink():raise ValueError('source evidence does not follow symlinks')
            if not path.is_file():continue
            count+=1;total+=path.stat().st_size
            if count>100000 or total>2*1024**3:raise ValueError('source evidence exceeds manifest bounds')
            digest.update(path.relative_to(source).as_posix().encode()+b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    git={}
    try:
        commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
        status=subprocess.run(['git','-C',str(source),'status','--porcelain','--untracked-files=normal'],capture_output=True,text=True,timeout=10)
        if commit.returncode==0:git=dict(commit=commit.stdout.strip(),dirty=bool(status.stdout))
    except (FileNotFoundError,subprocess.TimeoutExpired):pass
    return dict(input_hash=digest.hexdigest() if inputs else None,files=count,bytes=total,**git)
