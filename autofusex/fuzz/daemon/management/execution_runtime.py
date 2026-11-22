"""Query the explicitly registered execution interpreter without importing Torch."""
import json
import os
import subprocess


def facts(store):
    with store.connect() as db:
        row=db.execute("SELECT manifest FROM environments WHERE id='local' AND state='ready'").fetchone()
    if not row:
        return None
    manifest=json.loads(row['manifest'])
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(manifest['env'])
    script = ('import json,sys; from autofusex.fuzz.client.environment import collect; '
              'from autofusex.fuzz.client.scheduling.devices import device_inventory_info; '
              'print(json.dumps(dict(environment=collect(sys.argv[1]),inventory=device_inventory_info())))')
    result=subprocess.run([manifest['python'],'-c',script,str(store.home)],env=env,
                          capture_output=True,text=True,timeout=45,check=True)
    value=json.loads(result.stdout.splitlines()[-1])
    return value
