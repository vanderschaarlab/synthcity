# flake8: noqa
# mypy: ignore-errors

# stdlib
import argparse
import ast
import importlib
import inspect
import io
import os
import re
import runpy
import shutil
import sys
from collections.abc import Callable, Iterator
from itertools import islice, product
from numbers import *
from typing import Any, Optional, Union

# third party
import cloudpickle

TYPE_MAP = {  # maps of type annotations
    Integral: int,
    Real: float,
    Complex: complex,
    object: Any,
}

# MOD_MAP = {  # maps module names to their common aliases
#     'numpy': 'np',
#     'pandas': 'pd'
# }


def get_type(x):
    """
    Examples:
    >>> get_type(None)
    >>> get_type([])
    list
    >>> get_type([1, 2, 3])
    list[int]
    >>> get_type([1, 'a'])
    list
    >>> get_type(dict(a=0.9, b=0.1))
    dict[str, float]
    >>> get_type(dict(a=0.9, b='a'))
    dict[str, typing.Any]
    >>> get_type({1, 2.0, None})
    set[typing.Optional[float]]
    >>> get_type(str)
    type
    >>> get_type(True)
    bool
    >>> get_type((1, 2.0))
    tuple[int, float]
    >>> get_type(tuple(range(9)))
    tuple[int, ...]
    >>> get_type(iter(range(9)))
    typing.Iterator[int]
    >>> get_type((i if i % 2 else None for i in range(9)))
    typing.Iterator[typing.Optional[int]]
    """

    def dispatch(T, *xs, maxlen=5):
        xs = [list(map(get_type, l)) for l in xs]
        if not xs or min(map(len, xs)) == 0:  # empty collection
            return T
        ts = tuple(map(get_common_suptype, xs))
        if len(ts) == 1:
            t = ts[0]
        elif len(ts) > maxlen:
            t = get_common_suptype(ts)
        else:
            t = ts
        if t is object:
            return T
        elif len(ts) > maxlen:
            return T[t, ...]
        else:
            return T[t]

    if x is None:
        return None
    if inspect.isfunction(x) or inspect.ismethod(x):
        return Callable
    for t in (list, set, frozenset):
        if isinstance(x, t):
            return dispatch(t, x)
    if isinstance(x, tuple):
        return dispatch(tuple, *[[a] for a in x], maxlen=4)
    if isinstance(x, dict):
        return dispatch(dict, x.keys(), x.values())
    if isinstance(x, io.IOBase):
        return type(x)
    if isinstance(x, Iterator):  #! may be too general
        return dispatch(Iterator, islice(x, 10))
    if isinstance(x, bool):
        return bool
    if isinstance(x, Integral):
        return Integral
    if isinstance(x, Real):
        return Real
    if isinstance(x, Complex):
        return Complex
    return type(x)


def get_suptypes(t):
    def suptypes_of_subscripted_type(t):
        T = t.__origin__
        args = t.__args__
        sts = [
            T[ts]
            for ts in product(*map(get_suptypes, args))
            if not all(t in (object, ...) for t in ts)
        ]
        return sts + get_suptypes(T)

    if inspect.isclass(t) and issubclass(t, type):
        sts = list(t.__mro__)
    elif hasattr(t, "__origin__"):
        sts = suptypes_of_subscripted_type(t)
    elif isinstance(t, type):
        sts = list(t.mro())
    elif t == Ellipsis:
        sts = [t]
    else:  # None, Callable, Iterator, etc.
        sts = [t, object]
    return sts


def get_common_suptype(ts, type_map=None):
    """Find the most specific common supertype of a collection of types."""
    ts = set(ts)
    assert ts, "empty collection of types"

    optional = any(t is None for t in ts)
    ts.discard(None)

    if not ts:
        return None

    sts = [get_suptypes(t) for t in ts]
    for t in min(sts, key=len):
        if all(t in ts for ts in sts):
            break
    else:
        return Any

    if type_map:
        t = type_map.get(t, t)
    if optional:
        t = Optional[t]
    return t


def test():
    def get_anno(xs):
        return get_common_suptype(map(get_type, xs))

    recs = [
        [None, 1, 1.2],
        [{1: 2}, {1: 2.2}, {1: 2.1, 3: 4}],
        [(x for x in range(10)), iter(range(10))],
    ]
    for xs in recs:
        print(get_anno(xs))


def get_full_name(x, global_vars={}):
    """
    Examples:
    >>> import numpy as np
    >>> G = lambda: {id(v): k for k, v in globals().items() if k[0] != '_'}
    >>> get_full_name(np.ndarray, G())
    'np.ndarray'
    >>> import scipy as sp
    >>> get_full_name(sp.sparse.csr_matrix, G())
    'sp.sparse.csr_matrix'
    >>> import scipy.sparse as sps
    >>> get_full_name(sparse.csr_matrix, G())
    'sps.csr_matrix'
    """

    def get_name(x):
        if x.__module__ == "typing":
            return x._name
        return getattr(x, "__qualname__", x.__name__)

    if x is Ellipsis:
        return "..."
    if x is None:
        return "None"
    if id(x) in global_vars:
        return global_vars[id(x)]
    if x.__module__ == "builtins":
        return x.__name__
    # handle the subscripted types
    if hasattr(x, "__origin__"):
        T, args = x.__origin__, x.__args__
        if T is Union and len(args) == 2 and args[1] is type(None):
            T, args = Optional, args[:1]
        T = get_full_name(T, global_vars)
        args = ", ".join(get_full_name(a, global_vars) for a in args)
        return f"{T}[{args}]"
    # find the module alias
    names = (f"{x.__module__}.{get_name(x)}").split(".")[::-1]
    mods = [importlib.import_module(names[-1])]
    print(names)
    for name in names[-2::-1]:
        print(name, mods[-1])
        mods.append(getattr(mods[-1], name))
    mods = mods[::-1]
    # find the first module that is imported
    for i, (name, mod) in enumerate(zip(names, mods)):
        if id(mod) in global_vars:
            names = names[:i] + [global_vars[id(mod)]]
            mods = mods[: i + 1]
            break
    # skip useless intermediate modules
    for k in range(1, len(names)):
        if k >= len(names) - 1:
            break
        for i, (name, mod) in enumerate(zip(names, mods)):
            if i + 1 + k >= len(names):
                break
            if hasattr(mods[-k], name):
                names = names[: i + 1] + names[-k:]
                mods = mods[: i + 1] + mods[-k:]
                break
    return ".".join(names[::-1])


def profiler(frame, event, arg):
    if event in ("call", "return"):
        filename = os.path.abspath(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        if filename.endswith(".py") and funcname[0] != "<" and CWD in filename:
            recs = TYPE_RECS.setdefault(filename, {})
            if "globals" not in recs:
                recs["globals", None] = {
                    id(v): k for k, v in frame.f_globals.items() if k[0] != "_"
                }
            if event == "call":
                # print(filename, funcname, frame.f_lineno, frame.f_locals)
                arg_types = {var: get_type(val) for var, val in frame.f_locals.items()}
                lineno = frame.f_lineno
            else:
                arg_types = {"return": get_type(arg)}
                #! assumes no nested function has the same name as the outer function
                lineno = max(
                    ln for ln, fn in recs if fn == funcname and ln <= frame.f_lineno
                )
            rec = recs.setdefault((lineno, funcname), {})
            for k, v in arg_types.items():
                rec.setdefault(k, []).append(v)
    return profiler


# *** run the script N times to collect type records ***

parser = argparse.ArgumentParser()
parser.add_argument("script", help="the script to run")
parser.add_argument("-n", type=int, default=1, help="number of times to run the script")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(
    "-i", action="store_true", help="prompt before overwriting each script"
)
parser.add_argument(
    "--log", default="type_records.pkl", help="output file for type records"
)
parser.add_argument("--cwd", default=None, help="working directory")
parser.add_argument(
    "--backup", action="store_true", help="backup the scripts before annotating them"
)

ARGS = parser.parse_args()
DIR = os.path.dirname(os.path.abspath(ARGS.script))
CWD = ARGS.cwd or DIR

try:
    TYPE_RECS = cloudpickle.load(open(ARGS.log, "rb"))
except:
    TYPE_RECS = {}  # {filename: {(lineno, funcname): {argname: [type]}}}}

sys.path.extend([DIR, CWD])
sys.setprofile(profiler)

for _ in range(ARGS.n):
    runpy.run_path(sys.argv[1], run_name="__main__")

sys.setprofile(None)

with open(ARGS.log, "wb") as f:
    cloudpickle.dump(TYPE_RECS, f)


# *** determine the type annotations from the type records ***


def get_type_annotations(type_records=TYPE_RECS):
    def recurse(x):
        if isinstance(x, dict):
            return {k: recurse(v) for k, v in x.items()}
        elif isinstance(x, list):
            return get_common_suptype(x, type_map=TYPE_MAP)
        else:
            return x

    return recurse(type_records)


annotations = get_type_annotations()

# if ARGS.verbose:
#     for path, recs in annotations.items():
#         print(path)
#         for (lineno, funcname), arg_types in recs.items():
#             print(f'  {funcname} (Ln{lineno}):')
#             print('    ' + ', '.join(f'{k}: {get_full_name(v)}' for k, v in arg_types.items()))


# *** write the type annotations to the script ***


def find_defs_in_ast(tree):
    def recurse(node):  # should be in order
        if isinstance(node, ast.FunctionDef):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from recurse(child)

    return list(recurse(tree))


def annotate_def(def_node: ast.FunctionDef, annotations) -> bool:
    key = (def_node.lineno, def_node.name)
    if key not in annotations:
        return False  # no type records for this function
    annos = annotations[key]
    A = def_node.args
    all_args = A.posonlyargs + A.args + A.kwonlyargs
    defaults = dict(zip(A.args + A.kwonlyargs, A.defaults + A.kw_defaults))
    all_args.extend(filter(None, [A.vararg, A.kwarg]))
    changed = False
    global_vars = annotations["globals", None]
    for a in all_args:
        if a.annotation is None and a.arg != "self":
            t = annos[a.arg]
            if a == A.vararg:
                if t is tuple:
                    t = Any
                else:
                    assert t.__origin__ is tuple
                    if (
                        len(t.__args__) == 1
                        or len(t.__args__) == 2
                        and t.__args__[1] is Ellipsis
                    ):
                        t = t.__args__[0]
                    else:
                        t = get_common_suptype(t.__args__)
            elif a == A.kwarg:
                assert t.__origin__ is dict
                t = t.__args__[1]
            if t is None:
                t = Any
            if a.arg in defaults:
                t = Union[t, get_type(defaults[a.arg])]
            anno = get_full_name(t, global_vars)
            a.annotation = ast.Name(anno)
            changed = True
    if def_node.returns is None:
        if "return" not in annos:
            print("No return type for", key, annos)
            exit()
        anno = get_full_name(annos["return"], global_vars)
        def_node.returns = ast.Name(anno)
        def_node.returns.lineno = max(a.lineno for a in all_args)
        changed = True
    return changed


def annotate_script(filepath, verbose=ARGS.verbose):
    s = open(filepath, encoding="utf8").read()
    lines = s.splitlines()
    defs = [
        d
        for d in find_defs_in_ast(ast.parse(s))
        if annotate_def(d, annotations[filepath])
    ]
    if not defs:
        return None
    if verbose:
        print("Adding annotations to", filepath, "\n")
    starts, ends, sigs = [], [], []
    for node in defs:
        ln0, ln1 = node.lineno, node.body[0].lineno
        starts.append(ln0 - 1)
        ends.append(ln1 - 1)
        node.body = []  # only keep signature
        line = re.match(r"\s*", lines[ln0 - 1])[0] + ast.unparse(
            node
        )  # keep indentation
        sigs.append(line)
        if verbose:
            print("Old:", *lines[ln0 - 1 : ln1 - 1], sep="\n")
            print(">" * 50)
            print("New:", sigs[-1], sep="\n")
            print("-" * 50)
    new_lines = []
    for s, e, sig in zip([None] + ends, starts + [None], sigs + [None]):
        new_lines.extend(lines[s:e])
        if sig is not None:
            new_lines.append(sig)
    return "\n".join(new_lines)


for path in annotations:
    s = annotate_script(path)
    if s is None:
        continue
    if ARGS.backup:
        shutil.copy(path, path + ".bak")
    if not ARGS.i or input(f"Overwrite {path}?").lower() == "y":
        with open(path, "w", encoding="utf8") as f:
            f.write(s)
