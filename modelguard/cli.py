"""CLI: scan <files> | sign <dir> --key K | verify <dir> --key K"""
import argparse
import sys

from .integrity import sign_dir, verify_dir
from .pickle_scan import scan_file


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="modelguard")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("files", nargs="+")
    for n in ("sign", "verify"):
        x = sub.add_parser(n)
        x.add_argument("dir")
        x.add_argument("--key", required=True)
    a = p.parse_args(argv)
    if a.cmd == "scan":
        bad = 0
        for f in a.files:
            res = scan_file(f)
            print(f"{f}: " + ("clean" if not res else f"{len(res)} finding(s)"))
            for r in res:
                print("  ", r)
            bad += any(r.severity == "critical" for r in res)
        return 1 if bad else 0
    if a.cmd == "sign":
        print(f"signed {len(sign_dir(a.dir, a.key.encode())['files'])} files")
        return 0
    probs = verify_dir(a.dir, a.key.encode())
    print("intact" if not probs else "\n".join(probs))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
