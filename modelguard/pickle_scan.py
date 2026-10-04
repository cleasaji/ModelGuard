"""Static scanner for pickle-based model files (.pkl, .joblib, PyTorch .pt/.pth zips).

Loading a pickle can execute arbitrary code: any ``GLOBAL``/``STACK_GLOBAL`` + ``REDUCE`` pair calls a function chosen
by the file's author. This scanner walks the opcode stream *without executing it* and reports every callable the
pickle would import, flagging known-dangerous ones. It is heuristic (deny-list + unknown-module warning): prefer
``safetensors`` for real distribution of weights.
"""
from __future__ import annotations

import io
import pickle
import pickletools
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

DANGEROUS_MODULES = {
    "os", "posix", "nt", "subprocess", "socket", "shutil", "runpy", "pty", "ctypes", "importlib", "webbrowser",
    "urllib", "urllib.request", "http.client", "ftplib", "smtplib", "telnetlib", "requests", "asyncio", "sys",
    "code", "codeop", "tempfile", "multiprocessing", "threading", "commands", "pdb", "builtins_eval",
}
DANGEROUS_NAMES = {
    "builtins": {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "delattr", "input",
                 "globals", "locals", "vars"},
    "__builtin__": {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "execfile", "input"},
    "marshal": {"loads", "load"}, "pickle": {"loads", "load", "_loads", "Unpickler"},
    "types": {"FunctionType", "CodeType", "LambdaType"}, "platform": {"popen"},
    "base64": {"b64decode", "decodebytes"}, "codecs": {"decode", "open"}, "operator": {"attrgetter", "methodcaller"},
}
KNOWN_SAFE_PREFIXES = ("numpy", "torch", "collections", "sklearn", "scipy", "pandas", "_codecs", "copyreg",
                       "joblib.numpy_pickle", "xgboost", "lightgbm", "transformers")
SAFE_BUILTINS = {"set", "frozenset", "slice", "complex", "bytearray", "range", "dict", "list", "tuple", "int",
                 "float", "str", "bool", "object", "bytes"}
STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}


@dataclass(frozen=True)
class Finding:
    severity: str            # critical | medium
    callable: str            # module.name
    offset: int
    message: str
    member: str = ""         # file inside a zip archive, if any

    def __str__(self):
        loc = f"{self.member}@{self.offset}" if self.member else f"@{self.offset}"
        return f"{self.severity.upper():<8} {self.callable:<32} {loc:<18} {self.message}"


def classify(module: str, name: str):
    root = module.split(".")[0]
    if module in DANGEROUS_MODULES or root in DANGEROUS_MODULES:
        return "critical", "imports a module that can run commands / open sockets / touch the filesystem"
    if name in DANGEROUS_NAMES.get(module, ()):
        return "critical", "calls a function that executes or reconstructs arbitrary code"
    if module in ("builtins", "__builtin__") and name in SAFE_BUILTINS:
        return None
    if module.startswith(KNOWN_SAFE_PREFIXES) or module.split(".")[0] in KNOWN_SAFE_PREFIXES:
        return None
    return "medium", "references a non-allow-listed class; its __reduce__/__setstate__ runs at load time"


def scan_bytes(data: bytes, member: str = "") -> List[Finding]:
    out: List[Finding] = []
    strs: List[Optional[str]] = []
    memo: Dict[int, Optional[str]] = {}
    nmemo = 0

    def emit(module, name, pos):
        c = classify(module, name)
        if c:
            out.append(Finding(c[0], f"{module}.{name}", pos, c[1], member))
    try:
        for op, arg, pos in pickletools.genops(data):
            n = op.name
            if n in STRING_OPS:
                strs.append(arg if isinstance(arg, str) else arg.decode("utf-8", "replace"))
            elif n == "MEMOIZE":
                memo[nmemo] = strs[-1] if strs else None
                nmemo += 1
            elif n in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = strs[-1] if strs else None
            elif n in ("GET", "BINGET", "LONG_BINGET"):
                strs.append(memo.get(arg))
            elif n in ("GLOBAL", "INST"):
                module, _, name = str(arg).partition(" ")
                emit(module, name, pos)
            elif n == "STACK_GLOBAL":
                if len(strs) >= 2 and strs[-2] and strs[-1]:
                    emit(strs[-2], strs[-1], pos)
                else:
                    out.append(Finding("medium", "<unresolved>", pos,
                                       "STACK_GLOBAL with a computed module/name (obfuscation?)", member))
    except Exception as e:                       # truncated / corrupt stream: report, don't crash
        out.append(Finding("medium", "<parse-error>", 0, f"could not fully parse pickle stream: {e}", member))
    return out


def scan_file(path: str) -> List[Finding]:
    if zipfile.is_zipfile(path):                 # PyTorch checkpoints are zip archives holding data.pkl
        found: List[Finding] = []
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.filename.endswith((".pkl", ".pickle")):
                    found += scan_bytes(z.read(info), member=info.filename)
        return found
    with open(path, "rb") as fh:
        data = fh.read()
    if path.endswith(".npy") and b"'descr'" in data[:200] and b"|O" in data[:200]:
        return [Finding("medium", "numpy.object_array", 0, ".npy holds an object array (pickled); "
                                                         "load with allow_pickle=False")]
    return scan_bytes(data)


class _Restricted(pickle.Unpickler):
    allow: Dict[str, Set[str]] = {}

    def find_class(self, module, name):
        if module in ("builtins", "__builtin__") and name in SAFE_BUILTINS:
            return super().find_class(module, name)
        if name in self.allow.get(module, ()) or "*" in self.allow.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked: {module}.{name} is not on the allow-list")


def safe_loads(data: bytes, allow: Optional[Dict[str, Iterable[str]]] = None):
    """Load a pickle that may only reference allow-listed globals (default: plain containers)."""
    u = _Restricted(io.BytesIO(data))
    u.allow = {m: set(n) for m, n in (allow or {}).items()}
    return u.load()
