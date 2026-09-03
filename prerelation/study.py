"""prerelation.study — the simulation frame.

One convention, so that a study is a data object rather than a script:

    run_study(config) -> tidy DataFrame, one row per replicate

The seed lives in the config, and the config is returned alongside the rows,
so a table can always be traced back to the run that produced it. Nothing
here is a result; the numbers a manuscript reports come from runs that are
logged when they are made.

Config keys
-----------
name        label carried into every row
model       "product"  Y = c(X) * U
            "min"      Y = min(c(X), T)
            "independent"  Y drawn independently of X
            "equivalence"  Y = X (+ optional noise)
ceiling     "identity" | ("linear", a) | ("power", p)
free        "uniform" | ("beta", a, b)
n           persons per replicate
reps        number of replicates
seed        base seed; replicate r uses seed + r
noise       optional SD of additive noise, clipped back to [0, 1]
delta       ceiling band width (default 0.05)
top_q       top-x quantile of the legitimacy check (default 0.8)
min_interior  absolute floor of the interior guard (default 10)

``top_q`` and ``min_interior`` are carried in the config, and into every
output row, for the same reason the seed is: a table swept over them is
otherwise not reproducible from its own contents. Their defaults are the
fixed conventions and any other value is outside the reported definition.
"""

from __future__ import annotations

import numpy as np

from .core import DELTA, MIN_INTERIOR, TOP_Q, direction, prereq_index

__all__ = ["EXAMPLE_CONFIG", "generate_pair", "run_study"]

EXAMPLE_CONFIG = {
    "name": "calibrated-product-identity",
    "model": "product",
    "ceiling": "identity",
    "free": "uniform",
    "n": 500,
    "reps": 20,
    "seed": 20260826,
    "noise": 0.0,
    "delta": DELTA,
    "top_q": TOP_Q,
    "min_interior": MIN_INTERIOR,
}


def _ceiling_of(spec):
    if spec in (None, "identity"):
        return lambda x: x
    kind = spec[0] if isinstance(spec, (tuple, list)) else spec
    if kind == "linear":
        a = float(spec[1])
        return lambda x: a * x
    if kind == "power":
        p = float(spec[1])
        return lambda x: x ** p
    raise ValueError(f"unknown ceiling spec {spec!r}")


def _free_draw(spec, n, rng):
    if spec in (None, "uniform"):
        return rng.uniform(0.0, 1.0, n)
    kind = spec[0] if isinstance(spec, (tuple, list)) else spec
    if kind == "beta":
        return rng.beta(float(spec[1]), float(spec[2]), n)
    raise ValueError(f"unknown free-component spec {spec!r}")


def generate_pair(config, rng):
    """Draw one (x, y) sample from the generating model in ``config``."""
    n = int(config["n"])
    model = config.get("model", "product")
    c = _ceiling_of(config.get("ceiling", "identity"))
    x = rng.uniform(0.0, 1.0, n)

    if model == "product":
        y = c(x) * _free_draw(config.get("free", "uniform"), n, rng)
    elif model == "min":
        y = np.minimum(c(x), _free_draw(config.get("free", "uniform"), n, rng))
    elif model == "independent":
        y = rng.uniform(0.0, 1.0, n)
    elif model == "equivalence":
        y = x.copy()
    else:
        raise ValueError(f"unknown model {model!r}")

    noise = float(config.get("noise", 0.0) or 0.0)
    if noise > 0:
        y = np.clip(y + rng.normal(0.0, noise, n), 0.0, 1.0)
    return x, y


def run_study(config):
    """Run one study configuration and return a tidy DataFrame.

    Every row is a replicate and carries the config fields needed to
    reproduce it, including the replicate seed.
    """
    import pandas as pd

    cfg = dict(EXAMPLE_CONFIG)
    cfg.update(config)
    reps = int(cfg["reps"])
    base_seed = int(cfg["seed"])
    delta = float(cfg.get("delta", DELTA))
    top_q = float(cfg.get("top_q", TOP_Q))
    min_interior = int(cfg.get("min_interior", MIN_INTERIOR))

    rows = []
    for r in range(reps):
        seed_r = base_seed + r
        rng = np.random.default_rng(seed_r)
        x, y = generate_pair(cfg, rng)
        res = prereq_index(x, y, delta, top_q=top_q, min_interior=min_interior)
        d, pi_fwd, pi_rev = direction(x, y, delta, top_q=top_q,
                                      min_interior=min_interior)
        rows.append(
            {
                "name": cfg["name"],
                "model": cfg.get("model", "product"),
                "ceiling": str(cfg.get("ceiling", "identity")),
                "free": str(cfg.get("free", "uniform")),
                "n": int(cfg["n"]),
                "rep": r,
                "seed": seed_r,
                "delta_band": delta,
                "top_q": top_q,
                "min_interior": min_interior,
                "PI": res["PI"],
                "A1": res["A1"],
                "A2": res["A2"],
                "q": res["q"],
                "ell": res["ell"],
                "PI_reverse": pi_rev,
                "Delta": d,
            }
        )
    return pd.DataFrame.from_records(rows)
