import os
import sys

DEFAULTS = {
    "machine": {
        "concurrency": None, "per_run_cpus": 2, "per_run_ram_gb": 6,
        "max_ram_gb": None, "max_cpus": None, "shard": None, "range": None,
        "run_retries": 2,
    },
    "protocol": {
        "lv_peak_mmhg": 30.0, "rv_peak_mmhg": 10.0, "n_steps": 20, "loading": "linear",
        "rv_peaks_mmhg": [3.75, 7.5, 10.0, 15.0, 30.0],
        "dpdt_ratios": [0.5, 1.0, 2.0, 4.0],
    },
    "material": {
        "model": "ho", "a": 0.10, "b": 8.023,
        "af_reference": 20.0, "bf_reference": 10.0, "endo_deg": 60.0, "epi_deg": -60.0,
        "sweep": {"af_kpa": [8.0, 50.0], "bf": [6.0, 16.0],
                  "endo_deg": [45.0, 75.0], "epi_deg": [-75.0, -45.0]},
        "formulation": "compressible", "kappa_mpa": 0.003,
    },
    "cohort": {
        "n_base": 20, "sampling": "nested", "variants_per_parent": 20, "n_synthetic": None,
        "geometry": {"f_lv": [0.70, 1.35], "f_rv": [0.70, 1.35], "shift": [-0.15, 0.15]},
        "realism_screen": True,
    },
}

DECISIONS = {
    "protocol.rv_peak_mmhg": "RV peak pressure; sets the RV:LV ratio",
    "material.a": "reference isotropic term", "material.b": "reference isotropic term",
    "material.formulation": "strongest lever on the fitted exponent",
    "material.endo_deg": "fibre helix angle", "material.epi_deg": "fibre helix angle",
    "cohort.sampling": "how the cohort is drawn",
}


def _find_config():
    env = os.environ.get("STUDY_CONFIG")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, os.pardir, "STUDY_CONFIG.yaml"),
                 os.path.join(here, "STUDY_CONFIG.yaml"),
                 os.path.join(os.getcwd(), "STUDY_CONFIG.yaml")):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None or k not in out:
            out[k] = v
    return out


class Config:
    def __init__(self, data, path=None):
        self._d = data
        self.path = path

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(f"no config section or key {k!r}") from None
        return Config(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        return self.__getattr__(k)

    def get(self, dotted, default=None):
        cur = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def as_dict(self):
        return self._d

    def solver_args(self, geom, af=None, bf=None, endo=None, epi=None):
        m, p = self._d["material"], self._d["protocol"]
        return [
            "--geom", str(geom),
            "--a", str(m["a"]), "--b", str(m["b"]),
            "--af", str(m["af_reference"] if af is None else af),
            "--bf", str(m["bf_reference"] if bf is None else bf),
            "--endo", str(m["endo_deg"] if endo is None else endo),
            "--epi", str(m["epi_deg"] if epi is None else epi),
            "--formulation", str(m["formulation"]),
            "--kappa", str(m["kappa_mpa"]),
            "--lvedp", str(p["lv_peak_mmhg"]),
            "--rvedp", str(p["rv_peak_mmhg"]),
            "--steps", str(p["n_steps"]),
        ]

    def echo(self, stream=sys.stderr):
        src = self.path or "built-in defaults (no STUDY_CONFIG.yaml found)"
        print(f"[config] {src}", file=stream)
        for dotted, why in DECISIONS.items():
            print(f"[config]   {dotted} = {self.get(dotted)!r}   <- {why}", file=stream)


def load(path=None):
    path = path or _find_config()
    if path is None:
        return Config(DEFAULTS, None)
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            f"[config] {path} exists but PyYAML is not installed, so it cannot be read.\n"
            f"         Refusing to fall back to defaults -- an edited pressure or material would be\n"
            f"         silently ignored and the cohort would be run at the wrong settings.\n"
            f"         Fix:  pip install pyyaml    (the runner image already includes it)"
        )
    try:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    except Exception as e:
        raise SystemExit(f"[config] cannot parse {path}: {e}")
    if not isinstance(user, dict):
        raise SystemExit(f"[config] {path} did not parse to a mapping")
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        print(f"[config] WARNING: ignoring unknown top-level section(s) in {path}: {sorted(unknown)}",
              file=sys.stderr)
        user = {k: v for k, v in user.items() if k in DEFAULTS}
    return Config(_merge(DEFAULTS, user), path)


CFG = load()

if __name__ == "__main__":
    import json
    CFG.echo(sys.stdout)
    print(json.dumps(CFG.as_dict(), indent=2))
