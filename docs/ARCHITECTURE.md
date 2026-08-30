# Architecture

## The one-line argument

A vehicle line fails in two coupled ways — flow (bottlenecks) and quality
(defects) — and both share a root cause: **the latency between drift and
detection**. Most plants watch them on two dashboards. This twin runs both
engines on one event log.

The event log is the design constraint. Some stations stream hundreds of
OPC-UA tags; manual and legacy bays stream nothing. A twin that only works
where data is rich is a twin of the easy half of the plant. So everything is
built on the one signal every station emits: **the timestamp of a part entering
and leaving**.

## Data flow

```
                    ┌──────────────────────────────┐
                    │   simulator.py (SimPy DES)   │
                    │  35 stations, buffers,       │
                    │  blocking, starving,         │
                    │  breakdowns, variants,       │
                    │  rework, correlated tags     │
                    └───────────┬──────────────────┘
                                │
              ┌─────────────────┴───────────────────┐
              │                                     │
     events.parquet                        truth_*.parquet
     tags.parquet                          (bottleneck, defects,
              │                             drift, episodes)
              │                                     │
      ┌───────┴────────┐                            │
      │ adapters.py    │                            │
      │ SimAdapter     │                            │
      │ BoschAdapter   │                            │
      └───────┬────────┘                            │
              │                                     │
      ┌───────┴─────────┐                           │
      │ reconstruct.py  │  timestamps ──▶ buffer    │
      │                 │  levels, occupancy,       │
      │                 │  blocked / starved /      │
      │                 │  active intervals         │
      └───┬─────────┬───┘                           │
          │         │                               │
   ┌──────┴───┐ ┌───┴──────┐                        │
   │detectors │ │  spc.py  │                        │
   │  FLOW    │ │ QUALITY  │                        │
   └──────┬───┘ └───┬──────┘                        │
          │         │                               │
          └────┬────┘                               │
               ▼                                    ▼
          predictions ──────────────▶ scoring.py ◀──┘
                                     (the ONLY module
                                      allowed to read truth)
```

`audit.py` enforces two boundaries and fails the build on either:

- **Backend** — no prediction module may reference a truth table.
- **Frontend** — no display component may present a numeric literal as a
  measurement. Every figure a user sees must come from the backend.

## Serving layer

```
LiveTwin  ──  precomputes detectors on a 30 s grid, serves snapshot(t)
   │
   ├── GET  /api/line                      static topology
   ├── GET  /api/snapshot?t=              one frame
   ├── GET  /api/timeline                 constraint migration + disruption log
   ├── GET  /api/escape-window            the split-screen comparison
   ├── GET  /api/instrumentation-ranking  where to spend the sensor budget
   ├── GET  /api/engines                  hazard, blind, coherence, conformal
   ├── GET  /api/coherence-series         WIP vs Little's Law over time
   ├── POST /api/counterfactual           Monte Carlo rollout (1-2 s)
   └── WS   /stream                       replay; accepts {t, speed, paused, blind}
```

Replay is an index lookup because the run is a fixed event log, so a 1 Hz
websocket costs almost nothing. Twins are cached per (run, blind set), which is
what makes the blind-station toggle instant: the second twin is built once and
then switched between.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `schema.py` | Frozen event schema; `LineAdapter` interface |
| `line_config.py` | 35 stations, tag specs, variants, defect physics, assumptions register |
| `injectors.py` | Bottleneck migration schedule; fixture coupling-loss drift |
| `simulator.py` | SimPy DES; writes events, tags, states, truth |
| `adapters.py` | `SimAdapter` (working), `BoschAdapter` (designed, not built) |
| `reconstruct.py` | Timestamps → buffer levels → blocked/starved/active |
| `detectors.py` | Bottleneck walk, active period, and two baselines |
| `spc.py` | Six control charts + threshold calibration to an alert budget |
| `scoring.py` | Episode scoring. Reads truth. Nothing else does. |
| `twin.py` | `LiveTwin`: detectors evaluated once on a 30 s grid, served by timestamp |
| `mpc.py` | Monte Carlo counterfactual: forks the line and simulates interventions |
| `hazard.py` | Defect risk at station k from stations 1..k only, with a leak check |
| `blind.py` | Virtual metrology, Gaussian-process posterior, Expected Information Gain |
| `coherence.py` | Little's Law self-audit |
| `conformal.py` | Distribution-free alert threshold |
| `forecast.py` | LSTM buffer forecaster: NumPy, BPTT, Adam |
| `graphsage.py` | GraphSAGE mean aggregator: NumPy, with an aggregation-off ablation |
| `engines.py` | Runs the six above; serves both the report and the API |
| `audit.py` | Mechanically enforces the truth boundary |
| `api/main.py` | FastAPI: REST endpoints plus a websocket replay stream |
| `web/` | React dashboard: four views over one twin |

## Why the simulator is the primary substrate

The headline claim is a **counterfactual**: N vehicles escaped without the twin,
M with it. That requires running the same line twice under identical conditions.
History happens once, so no archived dataset can supply it.

The same holds for every other metric. "Which station was the momentary
bottleneck at 14:32" is a derived quantity no MES records. ARL₁ needs the exact
instant drift began, knowable only if you injected it.

The simulator is a measurement instrument, not a stand-in for data we could not
get. A wind tunnel is not a weaker aeroplane.

Three defences against circularity:

1. **Structural** — truth tables are read by exactly one module, enforced by
   `audit.py`.
2. **Calibrated, not chosen** — defect physics constants are solved numerically
   against a Bosch-anchored target rate (`scripts/calibrate_defect_physics.py`).
   Every other parameter is listed in `docs/ASSUMPTIONS.md`.
3. **Asserted hard** — `scripts/test_invariants.py` fails the build if the drift
   ever becomes visible to a univariate chart.

## Seed protocol

| Seeds | Purpose | Drift |
|---|---|---|
| 7, 21, 22, 23 | Evaluation — every reported number | on |
| 11, 12 | SPC threshold calibration only | off |
| 31, 32 | Detector hyperparameter tuning only | on |

No seed is used for two purposes.

## Scope: implemented vs designed

Round 1 proposed nine engines. Under a three-day Round 2 deadline we built the
spine properly rather than stubbing all nine.

**Implemented and measured:** discrete-event twin core, state reconstruction
from timestamps, active-period detection, inventory-augmented bottleneck walk,
five control charts plus our T²-CUSUM, alert-budget calibration, episode scoring.

**Implemented since:** Monte Carlo MPC counterfactual, live replay backend,
five-view dashboard with the blind-station toggle, hazard model, virtual
metrology, Gaussian-process posterior, Expected Information Gain, Little's Law
self-audit, conformal calibration. Seven of the nine Round 1 engines.

All nine Round 1 engines are implemented. The two neural ones are hand-written
in NumPy with gradient checks against finite differences in the test suite.

**Still roadmap:** LLM copilot, Bosch validation.

Saying which is which is the honest version. Nine half-built engines would be
the other one.
