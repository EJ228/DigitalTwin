"""
FastAPI backend for the DigitalTwin.ai dashboard.

    uvicorn api.main:app --reload --port 8000

Runs are precomputed and served by timestamp, so replay is an index lookup and
a 1 Hz websocket costs almost nothing. Twins are cached per (run, blind set),
so toggling a station's sensors off builds a second twin once and then switches
between them instantly -- which is what makes the blind-station demo snappy.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dtwin.line_config import DRIFT_STATION, TAKT_SECONDS, build_line
from dtwin.coherence import check as coherence_check
from dtwin.engines import evaluate_all
from dtwin.mpc import rollout
from dtwin.twin import load_twin

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

app = FastAPI(title="DigitalTwin.ai", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo only; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


def _twin(run: str, blind: str = ""):
    run_dir = DATA / run
    if not run_dir.exists():
        raise HTTPException(404, f"run '{run}' not found. Generate it with run_all.py")
    stations = tuple(sorted(s for s in blind.split(",") if s))
    return load_twin(str(run_dir), stations)


@app.get("/api/health")
def health():
    return {"ok": True, "runs": sorted(p.name for p in DATA.glob("*") if p.is_dir())}


@app.get("/api/line")
def line():
    """Static topology. Fetched once by the client."""
    stations = build_line()
    return {
        "takt_seconds": TAKT_SECONDS,
        "drift_station": DRIFT_STATION,
        "zones": ["body", "paint", "final"],
        "stations": [
            {
                "id": s.station_id, "zone": s.zone, "index": i,
                "tier": s.tier.value, "manual": s.manual,
                "buffer_capacity": s.buffer_out,
                "tags": [t.name for t in s.tags],
            }
            for i, s in enumerate(stations)
        ],
    }


@app.get("/api/runs")
def runs():
    """Run picker contents. Only the time span is needed, and that is one
    parquet column -- building a twin per run here would cost minutes."""
    out = []
    for p in sorted(DATA.glob("*")):
        if (p / "events.parquet").exists():
            t = pd.read_parquet(p / "events.parquet", columns=["t"]).t
            t0, t1 = float(t.min()), float(t.max())
            out.append({"run": p.name, "t_start": t0, "t_end": t1,
                        "hours": round((t1 - t0) / 3600, 1)})
    return out


@app.get("/api/snapshot")
def snapshot(run: str = "run_s7", t: float = 0.0, blind: str = ""):
    tw = _twin(run, blind)
    return tw.snapshot(max(tw.t0 + 600.0, min(t, tw.t1)))


@app.get("/api/timeline")
def timeline(run: str = "run_s7", blind: str = ""):
    """Plant-manager view: bottleneck migration, disruption log, coverage."""
    tw = _twin(run, blind)
    return {
        "bottleneck_runs": tw.shift_timeline(),
        "disruptions": tw.disruption_log(),
        "coverage": tw.coverage(),
        "t_start": tw.t0, "t_end": tw.t1,
    }


@app.get("/api/escape-window")
def escape_window(run: str = "run_s7", blind: str = ""):
    """The split-screen comparison."""
    return _twin(run, blind).escape_window()


class CounterfactualRequest(BaseModel):
    run: str = "run_s7"
    t: float
    blind: str = ""
    replicates: int = 5


@app.post("/api/counterfactual")
def counterfactual(req: CounterfactualRequest):
    """Monte Carlo rollout of candidate interventions. Takes 1-2 seconds."""
    tw = _twin(req.run, req.blind)
    t = max(tw.t0 + 600.0, min(req.t, tw.t1 - 60.0))
    return rollout(tw, t, replicates=max(1, min(req.replicates, 12)))


@app.get("/api/instrumentation-ranking")
def instrumentation_ranking(run: str = "run_s7", blind: str = ""):
    """Leadership view: where to spend the sensor budget next.

    Superseded by /api/engines, which ranks by Expected Information Gain
    computed from an actual Gaussian-process posterior. This endpoint keeps the
    older heuristic (constraint share weighted by tier) for the leadership view
    and is labelled as a proxy wherever it is displayed.
    """
    tw = _twin(run, blind)
    counts = tw.pred_walk.predicted.value_counts(normalize=True).to_dict()
    weight = {"rich": 0.15, "sparse": 0.6, "blind": 1.0}
    rows = []
    for s in tw.stations:
        tier = "blind" if s.station_id in tw.blind else s.tier.value
        share = float(counts.get(s.station_id, 0.0))
        rows.append({
            "station": s.station_id, "zone": s.zone, "tier": tier,
            "constraint_share": round(share, 4),
            "observability_gap": weight[tier],
            "score": round(share * weight[tier], 5),
        })
    rows.sort(key=lambda r: -r["score"])
    return {"ranking": rows[:10], "coverage": tw.coverage()}


def _precomputed_engines(run: str) -> dict | None:
    """The completed offline evaluation, but only if it is for THIS run."""
    path = ROOT / "results" / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        engines = json.load(f).get("engines")
    if engines and engines.get("hazard", {}).get("tested_on") == run:
        return engines
    return None


# Runs whose on-demand evaluation has completed in this process. lru_cache does
# not expose its keys, so readiness is tracked here rather than inferred.
_EVALUATED: set[str] = set()


@app.get("/api/engines/availability")
def engines_availability(run: str = "run_s7"):
    """Is this run's evaluation ready, or would asking for it start a long job?

    Cheap: a file read plus a set lookup. The dashboard calls this before
    /api/engines so it can offer the work rather than silently hanging for
    minutes on a tab click.
    """
    if not (DATA / run).exists():
        raise HTTPException(404, f"run '{run}' not found")
    return {
        "run": run,
        "precomputed": _precomputed_engines(run) is not None,
        "cached": run in _EVALUATED,
    }


@app.get("/api/engines")
def engines(run: str = "run_s7"):
    """Every offline engine: hazard, blind, coherence, conformal, forecast, graphsage.

    Served from the same evaluation the reported numbers come from, so the
    dashboard cannot display a figure the pipeline did not produce. Cached:
    the first call trains the hazard model and takes a few seconds.
    """
    if not (DATA / run).exists():
        raise HTTPException(404, f"run '{run}' not found")

    # Fast path: results.json holds one completed evaluation, and run_all.py
    # computes it for run_s7. Serving it for any OTHER run would report run_s7's
    # numbers under that run's name, so the cached file is only used when the
    # run actually matches. Everything else is evaluated for real and memoised
    # by evaluate_all's lru_cache, so only the first request per run pays.
    cached = _precomputed_engines(run)
    if cached:
        return {**cached, "computed": "precomputed"}

    out = evaluate_all(str(DATA), run)
    _EVALUATED.add(run)
    return {**out, "computed": "on-demand"}


@app.get("/api/coherence-series")
def coherence_series(run: str = "run_s7"):
    """Little's Law: observed WIP against throughput x flow time, over time."""
    tw = _twin(run)
    r = coherence_check(tw.events, tw.ids)
    if not r.get("available"):
        return {"available": False}
    return {"available": True, "tolerance": r["tolerance"],
            "mean_abs_error": r["mean_abs_error"], "coherent": r["coherent"],
            "verdict": r["verdict"], "series": r["series"]}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Replay a run. Client may send {speed, t, paused, blind} at any time."""
    await ws.accept()
    run = ws.query_params.get("run", "run_s7")
    blind = ws.query_params.get("blind", "")
    try:
        tw = _twin(run, blind)
    except HTTPException as e:
        await ws.send_json({"error": e.detail})
        await ws.close()
        return

    t = float(ws.query_params.get("t", tw.t0 + 600.0))
    speed = float(ws.query_params.get("speed", 60.0))   # sim seconds per real second
    paused = False

    async def receive():
        nonlocal t, speed, paused, tw
        while True:
            msg = await ws.receive_json()
            if "t" in msg:
                t = max(tw.t0 + 600.0, min(float(msg["t"]), tw.t1))
            if "speed" in msg:
                speed = float(msg["speed"])
            if "paused" in msg:
                paused = bool(msg["paused"])
            if "blind" in msg:
                tw = _twin(run, str(msg["blind"]))

    task = asyncio.create_task(receive())
    try:
        while True:
            await ws.send_json(tw.snapshot(t))
            await asyncio.sleep(1.0)
            if not paused:
                t += speed
                if t >= tw.t1:
                    t = tw.t0 + 600.0
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
