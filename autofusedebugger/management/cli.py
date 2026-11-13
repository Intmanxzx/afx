"""Stable JSON CLI. Parsing and management never import model dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from .config import canonical, load, parse, validate
from .daemon import Manager, request
from .store import Store, TERMINAL


def parser():
    root=argparse.ArgumentParser(prog='autofusedbg')
    root.add_argument('group',choices=['daemon','device','environment','task','operation','config','build','tool','management'])
    root.add_argument('action')
    root.add_argument('value',nargs='?')
    root.add_argument('--data-dir')
    root.add_argument('--json',action='store_true')
    root.add_argument('--foreground',action='store_true')
    root.add_argument('--endpoint',default='https://autofuse.online')
    root.add_argument('--name',default=__import__('socket').gethostname()[:80])
    root.add_argument('--request-key')
    root.add_argument('--workers',type=int)
    root.add_argument('--timeout',type=float,default=300)
    root.add_argument('--graceful',action='store_true')
    root.add_argument('--token-stdin',action='store_true')
    root.add_argument('--after',type=int,default=0)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    from ..fuzz.config import data_home
    home=data_home(args.data_dir);store=Store(home)
    try:
        if args.group=='daemon':
            if args.action=='supervise':
                from .launcher import run
                return run(home)
            if args.action=='run' or (args.action=='start' and args.foreground):
                Manager(home).run();return 0
            if args.action=='start':
                try: result=request(home,dict(action='status'))
                except (OSError,ValueError):
                    log=(home/'manager-start.log').open('ab')
                    try:
                        child=subprocess.Popen([sys.executable,'-m','autofusedebugger','daemon','supervise','--data-dir',str(home)],
                                               start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                    finally: log.close()
                    for _ in range(100):
                        if child.poll() is not None: raise ValueError('daemon exited during startup')
                        try: result=request(home,dict(action='status'));break
                        except OSError: time.sleep(.05)
                    else: raise TimeoutError('daemon startup timed out')
            elif args.action=='status': result=request(home,dict(action='status'))
            elif args.action=='stop': result=request(home,dict(action='shutdown'))
            else: raise ValueError('unknown daemon action')
        elif args.group=='device' and args.action=='enroll':
            if not args.token_stdin: raise ValueError('use --token-stdin; tokens are never command-line arguments')
            token=sys.stdin.readline(258).strip()
            if not 20<=len(token)<=256: raise ValueError('invalid registration token')
            result=request(home,dict(action='enroll',endpoint=args.endpoint,name=args.name,token=token))
        elif args.group=='device' and args.action=='pair':
            result=request(home,dict(action='pair',endpoint=args.endpoint,name=args.name))
        elif args.group=='management' and args.action=='trust':
            if not args.value: raise ValueError('local trust file required')
            result=request(home,dict(action='trust',policy=parse(Path(args.value).read_text())))
        elif args.group=='config' and args.action=='effective': result=request(home,dict(action='effective'))
        elif args.group=='operation':
            if not args.value: raise ValueError('operation ID required')
            if args.action=='cancel':result=request(home,dict(action='submit',kind='operation.cancel',payload=dict(operation_id=args.value),request_key=args.request_key or str(uuid.uuid4())))
            elif args.action=='logs': result=request(home,dict(action='events',id=args.value,after=args.after))
            elif args.action in {'inspect','wait'}:
                started=time.monotonic()
                while True:
                    result=request(home,dict(action='operation',id=args.value))
                    if args.action=='inspect' or result['state'] in TERMINAL: break
                    if time.monotonic()-started>args.timeout: raise TimeoutError('operation wait timed out; operation still exists')
                    time.sleep(.2)
            else: raise ValueError('unknown operation action')
        else:
            kind=args.group+'.'+args.action
            if kind in {'task.submit','task.validate','environment.plan','config.apply'}:
                payload=load(args.value)
                if args.workers is not None:
                    if payload['kind']!='fuzz_task': raise ValueError('workers override is task-only')
                    payload['execution']['resources'].update(policy='manual',workers=args.workers)
                    payload=validate(payload)
            elif kind=='tool.stage': payload=parse(Path(args.value).read_text())
            elif kind=='tool.activate': payload=dict(release_id=args.value)
            elif kind=='environment.inspect': payload={}
            elif kind=='environment.apply': payload=dict(plan_id=args.value)
            elif args.group=='task' and args.action in {'pause','resume','stop','force-stop'}:
                kind='task.action';payload=dict(run_id=args.value,action=args.action)
            elif kind=='task.reconfigure': payload=parse(Path(args.value).read_text())
            elif kind=='build.submit': payload=parse(Path(args.value).read_text())
            else: raise ValueError('unsupported management action')
            result=request(home,dict(action='submit',kind=kind,payload=payload,request_key=args.request_key or str(uuid.uuid4())))
        print(canonical(dict(schema_version=1,ok=True,data=result)))
        return 1 if isinstance(result,dict) and result.get('state') in {'failed','needs_attention','cancelled'} else 0
    except (ValueError,TypeError,OSError,LookupError,PermissionError,TimeoutError) as exc:
        print(canonical(dict(schema_version=1,ok=False,error=dict(code=type(exc).__name__,message=str(exc)[:400]))))
        return 2
