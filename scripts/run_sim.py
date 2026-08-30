"""
Generate a dataset.

    python scripts/run_sim.py --seed 7 --shifts 3 --out data/run_s7

Writes:
    events.parquet          what any detector is allowed to read
    tags.parquet            process readings, missing at sparse/blind stations
    states.parquet          per-station timeline (simulator-only; for scoring)
    truth_bottleneck.parquet
    truth_defects.parquet
    truth_drift.parquet
    manifest.json           seed, config, assumptions, git-free reproducibility

Paired counterfactual runs use the same seed with --no-drift, which -- thanks
to common random numbers in the simulator -- reproduces identical part flow.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dtwin import line_config as cfg
from dtwin.simulator import (
    EOL_FALSE_POSITIVE,
    EOL_SENSITIVITY,
    PER_STATION_BACKGROUND,
    AssemblyLineSim,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shifts", type=float, default=3.0, help="8-hour shifts")
    ap.add_argument("--out", type=str, default="data/run")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--no-bottlenecks", action="store_true")
    ap.add_argument("--drift-onset-frac", type=float, default=0.55)
    args = ap.parse_args()

    horizon = args.shifts * 8 * 3600.0
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from dtwin.injectors import default_drift

    drift = None if args.no_drift else default_drift(t_onset=args.drift_onset_frac * horizon)

    t0 = time.time()
    sim = AssemblyLineSim(
        horizon=horizon,
        seed=args.seed,
        drift=drift,
        enable_drift=not args.no_drift,
        enable_bottlenecks=not args.no_bottlenecks,
    )
    frames = sim.run()
    elapsed = time.time() - t0

    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)

    stations = cfg.build_line()
    tier_counts: dict[str, int] = {}
    for s in stations:
        tier_counts[s.tier.value] = tier_counts.get(s.tier.value, 0) + 1

    completed = frames["events"].query(
        "station_id == @cfg.EOL_STATION and event_type == 'exit'"
    )
    manifest = {
        "seed": args.seed,
        "horizon_seconds": horizon,
        "shifts": args.shifts,
        "wall_seconds": round(elapsed, 2),
        "drift_enabled": not args.no_drift,
        "bottlenecks_enabled": not args.no_bottlenecks,
        "assumptions": {
            "takt_seconds": cfg.TAKT_SECONDS,
            "target_load_seconds": cfg.TARGET_LOAD_SECONDS,
            "n_stations": len(stations),
            "sensor_tiers": tier_counts,
            "under_instrumented_pct": round(
                100 * (tier_counts.get("sparse", 0) + tier_counts.get("blind", 0))
                / len(stations), 1),
            "cpk_sigma": 4.0,
            "gap_diff_tolerance_mm": cfg.GAP_DIFF_TOLERANCE_MM,
            "gap_diff_steepness": cfg.GAP_DIFF_STEEPNESS,
            "base_defect_rate": cfg.BASE_DEFECT_RATE,
            "per_station_background": PER_STATION_BACKGROUND,
            "eol_sensitivity": EOL_SENSITIVITY,
            "eol_false_positive": EOL_FALSE_POSITIVE,
        },
        "observed": {
            "parts_released": len(sim.parts),
            "parts_completed": int(len(completed)),
            "throughput_per_hour": round(len(completed) / (horizon / 3600), 2),
            "rows": {k: int(len(v)) for k, v in frames.items()},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {out}  ({elapsed:.1f}s)")
    print(json.dumps(manifest["observed"], indent=2))


if __name__ == "__main__":
    main()
