import { useEffect, useState } from "react";
import MetricTile from "../components/MetricTile";
import { getEscapeWindow, getRanking } from "../lib/api";

// Illustrative, and labelled as such on screen.
const REWORK_COST = 340;
const INCIDENTS_PER_YEAR = 26;

/** Leadership: should we roll this out. Money and risk, minimal jargon. */
export default function LeadershipView({ run, blind }) {
  const [ew, setEw] = useState(null);
  const [rk, setRk] = useState(null);
  useEffect(() => {
    getEscapeWindow(run, blind).then(setEw).catch(() => {});
    getRanking(run, blind).then(setRk).catch(() => {});
  }, [run, blind]);

  if (!ew || !rk) return <div className="h-[420px] card animate-pulse" />;

  const saved = (ew.baseline_flag_at ?? 0) - (ew.twin_flag_at ?? 0);
  const perIncident = saved * REWORK_COST;
  const annual = perIncident * INCIDENTS_PER_YEAR;
  const cov = rk.coverage;

  return (
    <div className="space-y-4">
      <div className="card px-8 py-7">
        <div className="label">Escape window &middot; cars built before the fault is caught</div>
        <div className="flex items-baseline gap-6 mt-3">
          <span className="font-mono text-6xl text-alert line-through decoration-2">
            {ew.baseline_flag_at}
          </span>
          <span className="text-3xl text-ink-faint">&rarr;</span>
          <span className="font-mono text-7xl text-accent">{ew.twin_flag_at}</span>
          <span className="text-xl text-ink-muted ml-2">
            a {ew.reduction_pct}% reduction
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile label="Cars saved &middot; per incident" value={saved} tone="good" />
        <MetricTile label="Value per incident" value={`$${(perIncident / 1000).toFixed(1)}k`}
                    delta={`at $${REWORK_COST}/rework`} />
        <MetricTile label="Annualised &middot; one line" value={`$${(annual / 1000).toFixed(0)}k`}
                    delta={`${INCIDENTS_PER_YEAR} incidents/yr`} tone="accent" />
        <MetricTile label="Line covered" value={cov.total}
                    delta={`${cov.under_instrumented_pct}% under-instrumented`} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="card p-5">
          <div className="text-lg font-medium">Sensor coverage</div>
          <div className="label mt-1 mb-5 leading-relaxed">
            The twin covers the whole line, including bays with no instrumentation at all.
          </div>
          {[
            ["rich", "Richly instrumented", "bg-state-run"],
            ["sparse", "Sparse sensors", "bg-state-blocked"],
            ["blind", "No sensors \u2014 manual bays", "bg-state-blind"],
          ].map(([k, label, cls]) => (
            <div key={k} className="mb-4">
              <div className="flex justify-between text-sm mb-1.5">
                <span>{label}</span>
                <span className="font-mono text-ink-muted">{cov.counts[k] ?? 0}</span>
              </div>
              <div className="h-2 bg-page rounded-full overflow-hidden">
                <div className={`h-full ${cls} rounded-full`}
                     style={{ width: `${(100 * (cov.counts[k] ?? 0)) / cov.total}%` }} />
              </div>
            </div>
          ))}
        </div>

        <div className="card p-5">
          <div className="text-lg font-medium">Instrument these next</div>
          <div className="label mt-1 mb-4 leading-relaxed">
            Ranked by how often a station binds, weighted by how little we observe there.
            A blind station that frequently constrains is worth the money; a well-instrumented
            one that never binds is not.
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-line">
                <th className="label font-normal pb-2">Station</th>
                <th className="label font-normal pb-2">Current sensing</th>
                <th className="label font-normal pb-2 text-right">Time as constraint</th>
              </tr>
            </thead>
            <tbody>
              {rk.ranking.slice(0, 6).map((r) => (
                <tr key={r.station} className="border-b border-line">
                  <td className="py-2 font-mono">{r.station}</td>
                  <td className={r.tier === "blind" ? "text-alert" : "text-ink-muted"}>
                    {r.tier === "blind" ? "none" : r.tier}
                  </td>
                  <td className="text-right font-mono">
                    {(100 * r.constraint_share).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="prose-note max-w-3xl">
        Cost figures use a stated assumption of ${REWORK_COST} per reworked unit and{" "}
        {INCIDENTS_PER_YEAR} drift incidents per line per year. Both are illustrative and
        should be replaced with the plant&rsquo;s own numbers before any investment case rests on them.
      </div>
    </div>
  );
}
