#!/usr/bin/env python3
"""Sign a canonical release manifest offline; never transmit or print the key."""
import argparse
import base64
import json
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--key',required=True)
    parser.add_argument('--key-id',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    keyfile=Path(args.key)
    if keyfile.stat().st_mode&0o077:raise ValueError('signing key must be private')
    manifest=json.loads(Path(args.manifest).read_text())
    key=load_pem_private_key(keyfile.read_bytes(),password=None)
    if not isinstance(key,Ed25519PrivateKey):raise ValueError('Ed25519 key required')
    payload=json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    envelope=dict(manifest=manifest,key_id=args.key_id,signature=base64.b64encode(key.sign(payload)).decode())
    with Path(args.out).open('x') as stream:json.dump(envelope,stream,indent=2)


if __name__=='__main__':main()
