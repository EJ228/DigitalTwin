# Assumptions register

Every number in this project is an assumption, not a measurement. They are all
listed here, each with its rationale, so a reviewer can challenge any single one.

| Parameter | Value | Rationale |
|---|---|---|
| Takt | 60 s | 60 units/hour, the rate the Round 1 deck assumes |
| Stations | 35 | Brief: "roughly 30-50 across body, paint, final" |
| Zone split | 12 body / 10 paint / 13 final | Typical automotive layout |
| Target station load | 50 s | 83% planned utilisation. At 92% every station runs continuously and there is no bottleneck to find; the question becomes ill-posed |
| Line balance | +/-2% | Real lines are balanced to a few percent. This is WHY the bottleneck shifts - no station is structurally dominant |
| Baseline bottleneck margin | 0.05 s | Consequence of the above: the baseline constraint is genuinely marginal |
| Buffers | 2-6 slots | Small inter-station banks, larger at zone boundaries and the paint oven |
| Sensor tiers | 22 rich / 7 sparse / 6 blind | Brief: majority well-instrumented, meaningful minority manual |
| Under-instrumented | 37% | Matches the Round 1 design envelope of ~40% |
| MTBF / MTTR scaling | x1.6 / x0.32 | Availability ~99.6%. Unscaled, random downtime swamps every injected episode and the bottleneck becomes whichever station is broken |
| Variants | A 50% / B 32% / C 18% | Mixed-model line; variant affects work content by zone |
| Cpk | 1.33 (+/-4 sigma spec) | Makes the drift scenario honest - a sub-sigma change cannot breach a 4-sigma limit |
| Gap correlation, healthy | r = +0.86 | Two locating pins move together under thermal growth and batch variation |
| Gap correlation, degraded | r = -0.10 | Fixture develops play. Marginals UNCHANGED; only the coupling breaks |
| Defect tolerance | 0.0950 mm | SOLVED, not chosen - see scripts/calibrate_defect_physics.py |
| Defect steepness | 100 /mm | Same solve |
| In-control defect rate | ~0.6-0.9% | Anchored to Bosch Production Line Performance's observed sub-1% |
| Full-drift defect rate | ~13% | High enough to matter, low enough that EOL does not catch unit one |
| Background per station | 8e-05 | So defects are not all attributable to one station |
| EOL sensitivity | 0.93 | Imperfect: some defects escape entirely, which is what warranty is |
| EOL false positive | 0.008 | Inspection is not free of type-I error either |
| Rework bays | 2, off-line | A flagged part is pulled off the line, not held on it |
| Alert budget | 1 false alarm / 1200 parts | 20 hours at 60 units/h. Deck's arithmetic: a 5% FP rate is one alert every 20 minutes and the team stops looking |
| Detector window | 1800 s | Grid-searched on held-out seeds 31, 32 |
| Detector weight | 0.6 | Same. Surface is flat from 0.3 to 1.5 |

| Batch latent loading | 0.55 gap, 0.35 clamp | Incoming panel quality shifts body-shop geometry tags together. Variance-preserving, so marginals are unchanged |
| Ambient latent loading | 0.62 temp, 0.58 humidity | Environmental conditions shift paint tags and weld current together |
| Ambient OU theta / sigma | 0.02 / 0.22 per part | Slow drift over a shift rather than white noise |
| Hazard train / test | 3 runs / 1 held out | One run holds one drift event, so a within-run split leaves no positives in train |
| VM rolling window | 20 parts | Per-part cycle times are conditionally independent; the answerable question is whether a bay is drifting slow |
| VM neighbour radius | 3 stations | Blocking and starvation propagate locally |
| GP observation noise | 1.0 s | Nominal sensor noise for the EIG computation |
| Conformal alpha | 1/1200 | One false alarm per 1200 parts, matching the alert budget |

## What we do not claim

* These are not measurements from a real plant. We had no access to one, and the
  brief does not expect it.
* The simulator is calibrated to be *plausible and hard*, not to reproduce any
  specific factory. Its job is to be an instrument on which detection latency is
  measurable, not a replica.
* Bosch Production Line Performance is referenced as a defect-rate anchor only.
  It is not used to validate the flow engine and could not be: it has no buffers,
  no blocking, no takt, and 6-minute timestamp granularity against a 60-second takt.

## Changing any of this

Parameters live in `dtwin/line_config.py` and `dtwin/simulator.py`. After
changing sigma, either correlation, or the target defect rates, re-run:

```bash
python scripts/calibrate_defect_physics.py   # re-solve the physics constants
python scripts/test_invariants.py           # confirm the scenario is still hard
```

If the invariant suite fails, the scenario has become trivial and every
downstream number is meaningless. That is the point of it.
