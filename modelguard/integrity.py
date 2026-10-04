"""Tamper-evident model manifests: SHA-256 per file + HMAC-SHA256 over the manifest."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Dict, List

MANIFEST = "MODEL_MANIFEST.json"


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: str) -> Dict[str, str]:
    out = {}
    for d, _, names in os.walk(root):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
            if rel != MANIFEST:
                out[rel] = _hash_file(os.path.join(d, n))
    return dict(sorted(out.items()))


def _mac(key: bytes, files: Dict[str, str]) -> str:
    return hmac.new(key, json.dumps(files, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def sign_dir(root: str, key: bytes) -> dict:
    files = _files(root)
    m = {"version": 1, "files": files, "hmac": _mac(key, files)}
    with open(os.path.join(root, MANIFEST), "w") as fh:
        json.dump(m, fh, indent=2)
    return m


def verify_dir(root: str, key: bytes) -> List[str]:
    """Returns a list of problems (empty list = intact)."""
    mp = os.path.join(root, MANIFEST)
    if not os.path.exists(mp):
        return ["manifest missing"]
    m = json.load(open(mp))
    problems = []
    if not hmac.compare_digest(m.get("hmac", ""), _mac(key, m["files"])):
        problems.append("manifest signature invalid (edited manifest or wrong key)")
    now = _files(root)
    for f, h in m["files"].items():
        if f not in now:
            problems.append(f"missing file: {f}")
        elif now[f] != h:
            problems.append(f"modified file: {f}")
    problems += [f"unexpected new file: {f}" for f in now if f not in m["files"]]
    return problems
