"""Simple user/token onboarding; token never enters result data or output.

The CLI accepts the token directly. Its durable private local storage is an
implementation detail needed for unattended upload, not a user-managed key file.
"""
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
from .config import data_home, setup, load_config
from .resources import adaptive_defaults

ENDPOINT='https://autofuse.online'


def initialize_user(home, token, interval=30, *, opener=urlopen, start_sync=True):
    if not isinstance(token,str) or not 20<=len(token)<=256 or any(ord(c)<33 for c in token):
        raise ValueError('invalid token')
    if not 1<=interval<=86400:
        raise ValueError('sync interval must be between 1 and 86400 seconds')
    request=Request(ENDPOINT+'/api/identity',headers={'Authorization':'Bearer '+token,'User-Agent':'AutofuseDebugger/0.1.2'})
    try:
        with opener(request,timeout=15) as response:
            raw=response.read(4097)
        if len(raw)>4096:raise ValueError('oversized identity')
        identity=json.loads(raw)
        import uuid
        uuid.UUID(identity['user_id'])
        user = identity['display_name']
        if not isinstance(user, str) or not user.strip() or len(user)>80:
            raise ValueError('invalid server identity')
    except (OSError,KeyError,TypeError,json.JSONDecodeError):
        raise ValueError('token verification failed; check network and administrator-issued identity') from None
    home=data_home(home);home.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,name=tempfile.mkstemp(prefix='.credential-',dir=home)
    try:
        with os.fdopen(fd,'w') as out:
            out.write(token+'\n');out.flush();os.fsync(out.fileno())
        credential=home/'upload.token';os.replace(name,credential)
        existing = load_config(home)
        budgets = adaptive_defaults(home) if existing is None else {
            key: existing[key] for key in ('cpu_workers', 'compile_threads',
                'memory_budget_bytes', 'cache_budget_bytes') if key in existing}
        policy = 'auto' if existing is None else existing.get('resource_policy', 'manual')
        setup(home,display_name=user,user_id=identity['user_id'],key_file=credential,resource_policy=policy,**budgets)
        from .sync.client import configure,start
        configure(home,ENDPOINT,credential,interval)
        result={'user':user,'server':ENDPOINT,'sync_interval':interval,'resources':budgets}
        if start_sync:result['sync']=start(home,home/'results/results.sqlite')
        return result
    finally:
        Path(name).unlink(missing_ok=True)
