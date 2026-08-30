"""Sanity checks on the simulator. Run before trusting any downstream model."""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/digitaltwin")

from dtwin.line_config import DRIFT_STATION, TAKT_SECONDS, build_line
from dtwin.schema import ACTIVE_STATES
from dtwin.simulator import AssemblyLineSim

pd.set_option("display.width", 160)


def main():
    t0 = time.time()
    sim = AssemblyLineSim(horizon=3 * 8 * 3600.0, seed=7)
    f = sim.run()
    print(f"sim wall time: {time.time() - t0:.1f}s")

    ev, tg, st = f["events"], f["tags"], f["states"]
    tb, td, tdr = f["truth_bottleneck"], f["truth_defects"], f["truth_drift"]

    print("\n=== VOLUME ===")
    print(f"events {len(ev):,}   tags {len(tg):,}   state intervals {len(st):,}")
    completed = ev[(ev.station_id == "S35") & (ev.event_type == "exit")]
    hours = sim.horizon / 3600
    print(f"parts released {len(sim.parts):,}   completed {len(completed):,}")
    print(f"throughput {len(completed)/hours:.1f} units/h   (takt implies 60.0)")
    print(f"effective takt {sim.horizon/max(1,len(completed)):.1f}s")

    print("\n=== STATION STATE MIX (top 8 by active fraction) ===")
    st = st.copy()
    st["dur"] = st.t_end - st.t_start
    piv = st.pivot_table(index="station_id", columns="state", values="dur",
                         aggfunc="sum").fillna(0.0)
    piv = piv.div(piv.sum(axis=1), axis=0)
    piv["active"] = sum(piv[c] for c in piv.columns if c in ACTIVE_STATES)
    print((piv.sort_values("active", ascending=False).head(8) * 100).round(1))

    print("\n=== TRUE BOTTLENECK TIMELINE ===")
    tb2 = tb.copy()
    tb2["blk"] = (tb2.true_bottleneck != tb2.true_bottleneck.shift()).cumsum()
    runs = tb2.groupby("blk").agg(
        station=("true_bottleneck", "first"),
        t_start=("t", "min"), t_end=("t", "max"),
        mean_margin=("margin", "mean"),
    )
    runs["hours"] = (runs.t_end - runs.t_start) / 3600
    print(runs[["station", "t_start", "t_end", "hours", "mean_margin"]].round(2).to_string(index=False))
    print(f"bottleneck shifts: {len(runs)-1} over {hours:.0f}h")

    print("\n=== DRIFT SCENARIO ===")
    print(tdr.to_string(index=False))
    onset = float(tdr.t_onset.iloc[0])

    s08 = tg[tg.station_id == DRIFT_STATION].pivot_table(
        index="part_id", columns="tag", values="value", aggfunc="first")
    s08_t = tg[tg.station_id == DRIFT_STATION].groupby("part_id").t.first()
    pre = s08[s08_t < onset]
    post = s08[s08_t > onset + 1800]
    print(f"S08 measured parts: pre-drift {len(pre)}, full-drift {len(post)}")

    print("\n--- spec breach rate per tag (%), 4-sigma limits ---")
    specs = {t.name: t for t in next(s for s in build_line()
                                     if s.station_id == DRIFT_STATION).tags}
    rows = []
    for nm, sp in specs.items():
        def br(df):
            v = df[nm]
            return 100.0 * ((v < sp.lsl) | (v > sp.usl)).mean()
        rows.append((nm, sp.lsl, sp.usl, br(pre), br(post)))
    br_df = pd.DataFrame(rows, columns=["tag", "LSL", "USL", "pre_%", "drift_%"])
    print(br_df.round(3).to_string(index=False))
    worst = br_df.drift_.max() if hasattr(br_df, "drift_") else br_df["drift_%"].max()
    print(f"-> worst univariate breach rate under full drift: {worst:.2f}%  "
          f"(must stay near in-control; if this is high the scenario is trivial)")

    print("\n--- multivariate signal ---")
    cols = list(specs)
    mu = pre[cols].mean().values
    cov = np.cov(pre[cols].values.T)
    inv = np.linalg.pinv(cov)
    def maha(df):
        d = df[cols].values - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))
    m_pre, m_post = maha(pre), maha(post)
    print(f"Mahalanobis  pre {m_pre.mean():.2f}   full drift {m_post.mean():.2f}")
    ucl = np.sqrt(np.percentile(m_pre**2, 99.73))
    print(f"T^2 single-point UCL (99.73% of in-control): {ucl:.2f}")
    print(f"single-point alarm rate under drift: {100*(m_post>ucl).mean():.1f}%"
          "  (low is GOOD -- proves accumulation is required)")

    gd_pre = (pre["s08_gap_left_mm"] - pre["s08_gap_right_mm"]).abs()
    gd_post = (post["s08_gap_left_mm"] - post["s08_gap_right_mm"]).abs()
    print(f"|gap_left - gap_right| mm:  pre {gd_pre.mean():.4f}   drift {gd_post.mean():.4f}")

    print("\n=== DEFECTS ===")
    td_done = td[td.part_id.isin(completed.part_id)]
    pre_p = td_done[td_done.t_release < onset]
    post_p = td_done[td_done.t_release > onset + 1800]
    print(f"overall defect rate {100*td_done.is_defective.mean():.2f}%")
    print(f"  pre-drift  {100*pre_p.is_defective.mean():.2f}%  (n={len(pre_p)})")
    print(f"  full drift {100*post_p.is_defective.mean():.2f}%  (n={len(post_p)})")
    print("\ncause mix:")
    print(td_done[td_done.is_defective].cause_station.value_counts().head(6))

    print("\n=== ESCAPE WINDOW (baseline: end-of-line inspection only) ===")
    s08_exit = (ev[(ev.station_id == DRIFT_STATION) & (ev.event_type == "exit")]
                .sort_values("t").reset_index(drop=True))
    s08_exit["build_pos"] = np.arange(len(s08_exit))
    pos = dict(zip(s08_exit.part_id, s08_exit.build_pos))

    # build position of the first part to pass S08 after drift onset
    after = s08_exit[s08_exit.t >= onset]
    if after.empty:
        print("drift onset falls outside the simulated horizon")
        return
    onset_pos = int(after.build_pos.iloc[0])

    # first part CAUSED by the drift (post-onset) that EOL actually flags
    caused = td[(td.cause_mechanism == "fixture_twist") & td.is_defective]
    caused = caused[caused.part_id.map(pos).fillna(-1) >= onset_pos]
    caught = caused[caused.detected_at.notna()]
    n_caused = len(caused)
    if caught.empty:
        print(f"{n_caused} drift-caused defects created; none reached EOL in horizon")
        return
    first = caught.sort_values("detected_at").iloc[0]
    first_pos = pos[int(first.part_id)]

    built = first_pos - onset_pos
    print(f"drift onset at t={onset:.0f}s -> S08 build position {onset_pos}")
    print(f"first drift-caused defect flagged at EOL: part {int(first.part_id)} "
          f"(S08 build position {first_pos})")
    print(f"ESCAPE WINDOW = {built} vehicles built at S08 before EOL flagged the drift")
    print(f"  of which drift-caused defective: "
          f"{int(caused[caused.part_id.map(pos) < first_pos].shape[0])}")
    lead = float(first.detected_at) - float(
        s08_exit.set_index('part_id').t.loc[int(first.part_id)])
    print(f"  S08 -> EOL transit for that part: {lead/60:.1f} min")
    print("\n(this is the number the twin has to beat; the Round-1 deck's ~17 "
          "was illustrative)")


if __name__ == "__main__":
    main()
