"""
Substrate adapters. One interface, several data sources.

The point of this file is that no model in the project knows or cares where its
event log came from. `SimAdapter` reads our generated runs; `BoschAdapter` reads
the Bosch Production Line Performance competition data. Both hand back the same
`LineSnapshot`, so the same detector runs on either.

Why the simulator is the primary substrate is argued in the README: our headline
claim is a counterfactual, and a counterfactual needs the same line run twice
under identical conditions. History only happens once.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from .line_config import build_line
from .schema import LineAdapter, LineSnapshot


class SimAdapter(LineAdapter):
    """Reads a run directory written by scripts/run_sim.py.

    `load()` returns events, tags and (simulator-only) true states.
    `truth()` is deliberately a SEPARATE method: models call load(), the scorer
    calls truth(). Keeping them apart at the interface is what makes "did you
    train on the labels?" a checkable question rather than a promise.
    """

    def __init__(self, run_dir: str | pathlib.Path):
        self.run_dir = pathlib.Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(
                f"{self.run_dir} not found. Generate it with:\n"
                f"  python scripts/run_sim.py --seed 7 --shifts 3 --out {self.run_dir}"
            )
        self.name = f"sim:{self.run_dir.name}"

    def _read(self, stem: str) -> pd.DataFrame:
        p = self.run_dir / f"{stem}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def load(self) -> LineSnapshot:
        import json

        manifest_path = self.run_dir / "manifest.json"
        meta = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return LineSnapshot(
            events=self._read("events"),
            tags=self._read("tags"),
            states=self._read("states"),
            meta=meta,
        )

    def truth(self) -> dict[str, pd.DataFrame]:
        """SCORER ONLY. Never call this from a detector."""
        return {
            "bottleneck": self._read("truth_bottleneck"),
            "defects": self._read("truth_defects"),
            "drift": self._read("truth_drift"),
            "episodes": self._read("truth_episodes"),
        }

    def stations(self) -> list[str]:
        return [s.station_id for s in build_line()]


class BoschAdapter(LineAdapter):
    """Bosch Production Line Performance (Kaggle, 2016).

    NOT IMPLEMENTED, and deliberately so. Bosch is a genuine reality check for
    the QUALITY model -- 1.18M real parts, 51 stations, sub-1% defect rate, and
    authentically filthy features. It is useless for the FLOW model: it has no
    buffers, no blocking or starving, no takt, and 6-minute timestamp
    granularity against our 60-second takt.

    We scoped it out rather than half-build it under a three-day deadline. The
    adapter interface exists so that adding it later is a single file and no
    change to any detector.

    Expected inputs: train_numeric.csv, train_date.csv, train_categorical.csv.
    Date columns are named L{line}_S{station}_D{feature}; station timestamps
    would be melted into the same ENTER/EXIT event schema used here.
    """

    def __init__(self, data_dir: str | pathlib.Path):
        self.data_dir = pathlib.Path(data_dir)
        self.name = "bosch"

    def load(self) -> LineSnapshot:
        raise NotImplementedError(
            "BoschAdapter is designed but not implemented -- see the roadmap in "
            "README.md. Use SimAdapter for all current results."
        )

    def stations(self) -> list[str]:
        raise NotImplementedError
