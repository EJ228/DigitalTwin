"""
Invariants the substrate must satisfy.

These are not unit tests of code paths. They are assertions that the SCENARIO
is still hard. It is easy to retune a constant and accidentally produce a drift
that a univariate chart catches instantly, at which point every downstream
result is meaningless. Run this before trusting any model output.

    python scripts/test_invariants.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dtwin.line_config import DRIFT_STATION, TAKT_SECONDS, build_line
from dtwin.simulator import AssemblyLineSim

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    stations = build_line()

    # ---- 1. the line is balanced -------------------------------------
    from dtwin.injectors import BottleneckSchedule
    flat = BottleneckSchedule([], stations)
    loads = np.array([flat.theoretical_load(s.station_id, 0.0) for s in stations])
    spread = (loads.max() - loads.min()) / loads.max()
    check("line balanced within 5%", spread < 0.05, f"spread {100*spread:.1f}%")
    check("no station exceeds takt at baseline", loads.max() < TAKT_SECONDS,
          f"max load {loads.max():.1f}s vs takt {TAKT_SECONDS}s")

    top2 = np.sort(loads)[-2:]
    check("baseline bottleneck is marginal (<0.5s over runner-up)",
          (top2[1] - top2[0]) < 0.5, f"margin {top2[1]-top2[0]:.2f}s")

    # ---- 2. sensor coverage matches the stated envelope ---------------
    tiers = pd.Series([s.tier.value for s in stations]).value_counts()
    under = (tiers.get("sparse", 0) + tiers.get("blind", 0)) / len(stations)
    check("under-instrumented share is 35-45%", 0.35 <= under <= 0.45,
          f"{100*under:.0f}% ({dict(tiers)})")

    # ---- 3. run the line ---------------------------------------------
    sim = AssemblyLineSim(horizon=3 * 8 * 3600.0, seed=7)
    f = sim.run()
    ev, tg, td = f["events"], f["tags"], f["truth_defects"]
    onset = float(f["truth_drift"].t_onset.iloc[0])

    completed = ev[(ev.station_id == "S35") & (ev.event_type == "exit")]
    tph = len(completed) / (sim.horizon / 3600)
    check("throughput is 55-68 units/h", 55 <= tph <= 68, f"{tph:.1f} u/h")

    # ---- 4. the bottleneck actually shifts within shifts -------------
    tb = f["truth_bottleneck"]
    runs = (tb.true_bottleneck != tb.true_bottleneck.shift()).sum() - 1
    per_shift = runs / (sim.horizon / (8 * 3600))
    check("bottleneck shifts >=3 times per shift", per_shift >= 3,
          f"{per_shift:.1f} shifts/shift")

    # ---- 5. THE CENTRAL CLAIM: no marginal distribution changes ------
    s08 = tg[tg.station_id == DRIFT_STATION]
    wide = s08.pivot_table(index="part_id", columns="tag", values="value", aggfunc="first")
    tmap = s08.groupby("part_id").t.first()
    pre = wide[tmap < onset]
    post = wide[tmap > onset + 1800]

    specs = {t.name: t for t in next(s for s in stations
                                     if s.station_id == DRIFT_STATION).tags}
    cols = list(specs)

    # every tag's mean and sd must be statistically indistinguishable
    worst_mean, worst_sd = 0.0, 0.0
    for nm in cols:
        sd = pre[nm].std()
        worst_mean = max(worst_mean, abs(post[nm].mean() - pre[nm].mean()) / sd)
        worst_sd = max(worst_sd, abs(post[nm].std() / sd - 1.0))
    check("no tag's MEAN moves more than 0.15 sigma", worst_mean < 0.15,
          f"worst {worst_mean:.3f} sigma -- a mean shift would be univariately visible")
    check("no tag's SD changes more than 15%", worst_sd < 0.15,
          f"worst {100*worst_sd:.1f}%")

    worst = max(
        100.0 * ((post[nm] < sp.lsl) | (post[nm] > sp.usl)).mean()
        for nm, sp in specs.items()
    )
    check("no univariate spec breach under full drift (<1%)", worst < 1.0,
          f"worst tag breaches {worst:.2f}% of parts")

    # the coupling, and only the coupling, must change
    r_pre = pre["s08_gap_left_mm"].corr(pre["s08_gap_right_mm"])
    r_post = post["s08_gap_left_mm"].corr(post["s08_gap_right_mm"])
    # The observed coupling does not fall to the nominal -0.10, and should not:
    # shared latents (incoming batch quality, ambient conditions) load on both
    # gaps, so part of their agreement survives the fixture developing play.
    # What must hold is that the coupling COLLAPSES -- that is the signal.
    check("gap coupling collapses by at least 0.5", r_pre - r_post > 0.5,
          f"r = {r_pre:.2f} -> {r_post:.2f}")

    mu, cov = pre[cols].mean().values, np.cov(pre[cols].values.T)
    inv = np.linalg.pinv(cov)

    def m2(df):
        d = df[cols].values - mu
        return np.einsum("ij,jk,ik->i", d, inv, d)

    check("the drift IS multivariately present",
          np.mean(m2(post)) > 1.4 * np.mean(m2(pre)),
          f"mean T^2 {np.mean(m2(pre)):.1f} -> {np.mean(m2(post)):.1f}")

    # ---- 6. defect rates are Bosch-plausible --------------------------
    done = td[td.part_id.isin(completed.part_id)]
    pre_rate = 100 * done[done.t_release < onset].is_defective.mean()
    post_rate = 100 * done[done.t_release > onset + 1800].is_defective.mean()
    check("in-control defect rate under 1.2%", pre_rate < 1.2, f"{pre_rate:.2f}%")
    check("drift lifts defect rate at least 5x", post_rate > 5 * pre_rate,
          f"{pre_rate:.2f}% -> {post_rate:.2f}%")

    # ---- 7. common random numbers hold across the counterfactual ------
    sim_off = AssemblyLineSim(horizon=3 * 8 * 3600.0, seed=7, enable_drift=False)
    f_off = sim_off.run()
    a = f["events"]; b = f_off["events"]
    a = a[a.event_type.isin(["enter", "exit"])].reset_index(drop=True)
    b = b[b.event_type.isin(["enter", "exit"])].reset_index(drop=True)
    check("paired drift-on/off runs share identical part flow", a.equals(b),
          f"{len(a)} vs {len(b)} flow events")

    # ---- 8. blind stations really are blind ---------------------------
    blind = {s.station_id for s in stations if s.tier.value == "blind"}
    leaked = set(tg.station_id.unique()) & blind
    check("blind stations emit zero process tags", not leaked, f"leaked: {leaked}")
    ts_ok = blind <= set(ev.station_id.unique())
    check("blind stations still emit timestamps", ts_ok)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} INVARIANT(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("all invariants hold")


if __name__ == "__main__":
    main()
