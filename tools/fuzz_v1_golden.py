"""Generate reviewed protocol vectors (explicit output only); never runtime cache."""
import hashlib
import json
from pathlib import Path
import sys

from autofusedebugger.fuzz.generation import materialize, workload_id, reference

records = []
for entropy in (0, 1, 42, 389609433, 2**128 - 1):
    token = str(entropy)
    bundle = materialize(token)
    records.append({"seed": token, "workload_id": workload_id(bundle),
                    "storage_sha256": [hashlib.sha256(v).hexdigest() for v in bundle.storages],
                    "oracle_sha256": [hashlib.sha256(reference(bundle, i)[0][1]).hexdigest()
                                      for i in range(len(bundle.spec.shape_samples))]})
Path(sys.argv[1]).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
