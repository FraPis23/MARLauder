"""Strict-JSON coercion, shared by the training driver, the web server and the analysis tools.

Deliberately dependency-free (stdlib only, no torch): `viz/web_server.py` keeps its import graph
torch-free on purpose, and `scripts/analyze_run.py` has to run on the host where numpy is not
installed. Anything heavier belongs in the caller.
"""
from __future__ import annotations

import math


def jsonable(obj):
    """Recursively coerce `obj` into something json.dumps renders as STRICT JSON.

    Two classes of value break strict JSON in this codebase, and both occur in practice:

    * Non-finite floats. `metric/steps_to_{50,90}*` is NaN whenever the coverage threshold is
      never crossed, `explore/ep_end` is NaN when no episode ended in the rollout, and
      `EnvCfg.revisit_streak_cap` defaults to +inf. Python's json.dumps emits the BARE tokens
      `NaN` / `Infinity` for these — `json.load` reads them back happily, but `JSON.parse`
      throws, which is why every `params.json` written so far and the dashboard's
      `/api/argschema` are invalid JSON to any non-Python reader. Mapping non-finite to `null`
      keeps the key present (a missing key and a null key mean different things to the analysis)
      while staying parseable everywhere.
    * Objects json does not know: Path, dataclass, tensor, ndarray, set.

    Note the bool check precedes the int check implicitly (bool IS an int in Python, and both are
    valid JSON scalars, so the shared branch is correct either way).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return jsonable(obj.item())              # 0-dim tensor / numpy scalar
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())            # tensor / ndarray
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return jsonable(vars(obj))
    return str(obj)
