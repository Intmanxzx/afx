"""Kernel-released exclusive lease keyed by stable NPU UUID, not logical index.

Default scope is all runs of the same OS user. Sites requiring cross-user
coordination must provision a shared protected directory and configure every
client with AUTOFUSEDBG_DEVICE_LOCK_DIR. This does not police external programs.
"""
import hashlib
import os
from pathlib import Path
import stat


class DeviceLease:
    def __init__(self, identity, directory=None):
        if not isinstance(identity,str) or not identity.strip() or len(identity)>256:
            raise ValueError('stable physical device identity required')
        self.identity=identity
        self.directory=Path(directory or os.environ.get('AUTOFUSEDBG_DEVICE_LOCK_DIR',
            str(Path.home()/'.local/state/autofusedbg/device-locks'))).expanduser().resolve()
        self.fd=None

    def acquire(self):
        if os.name!='posix':raise ValueError('NPU device leases currently require POSIX')
        import fcntl
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        name=hashlib.sha256(self.identity.encode()).hexdigest()+'.lock'
        fd=os.open(self.directory/name,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('lease inode must be a regular file')
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('device busy: another managed test owns this physical NPU') from None
            os.set_inheritable(fd,True)
            self.fd=fd
            return self
        except BaseException:
            os.close(fd);raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None
