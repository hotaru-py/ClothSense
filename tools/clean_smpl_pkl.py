"""Rewrite SMPL .pkl files without chumpy objects (chumpy does not import under numpy 2).

    .venv/Scripts/python tools/clean_smpl_pkl.py models/smpl/SMPL_NEUTRAL.pkl [...]

The official SMPL pickles store some arrays (shapedirs, ...) as chumpy `Ch` objects. We unpickle with stand-in
classes that just keep the pickled state, take each object's value array (`x`), and write a plain-numpy pickle
in place (the original is kept as *.orig.pkl). smplx and SMPLicit both load the cleaned file.
"""
import pickle
import sys
from pathlib import Path

import numpy as np


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _value(o, depth=0):
    if isinstance(o, _Stub):
        st = o.__dict__.get("_state", {})
        st = st[0] if isinstance(st, tuple) and st else st
        if isinstance(st, dict) and "x" in st:
            return _value(st["x"], depth + 1)
        if isinstance(st, dict) and "a" in st:              # some wrappers keep the operand as 'a'
            return _value(st["a"], depth + 1)
        raise ValueError(f"chumpy object without a stored value: keys {list(st)[:8] if isinstance(st, dict) else type(st)}")
    return o


def clean(path: Path) -> list:
    with open(path, "rb") as f:
        model = _Unpickler(f, encoding="latin1").load()
    changed = []
    for k, v in list(model.items()):
        if isinstance(v, _Stub):
            model[k] = np.asarray(_value(v))
            changed.append(k)
    if changed:
        orig = path.with_suffix(".orig.pkl")
        if not orig.exists():
            path.replace(orig)
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=2)
    return changed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "-> converted:", clean(Path(p)) or "nothing (already plain numpy)")
