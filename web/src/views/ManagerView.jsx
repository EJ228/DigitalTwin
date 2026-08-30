import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import MetricTile from "../components/MetricTile";
import { getTimeline } from "../lib/api";
import { pct } from "../lib/format";

// Measured on 4 seeds x 10 disruptions. Source: results/RESULTS.md.
const COMPARISON = [
  { name: "This twin", found: 100 },
  { name: "Active period only", found: 70 },
  { name: "Utilisation report", found: 45 },
  { name: "Longest queue", found: 25 },
];

/** Plant manager: how the shift went and what to change. */
export default function ManagerView({ snapshot, run, blind }) {
  const [data, setData] = useState(null);
  useEffect(() => { getTimeline(run, blind).then(setData).catch(() => {}); }, [run, blind]);

  if (!data) return <div className="h-[420px] card animate-pulse" />;
  const m = snapshot?.metrics;
  const span = data.t_end - data.t_start;

  return (
    <div className="space-y-4">
      {m && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricTile label="OEE" value={pct(m.oee, 1).replace("%", "")} unit="%"
                      progress={m.oee} tone="accent" />
          <MetricTile label="Availability" value={pct(m.availability, 1).replace("%", "")} unit="%"
                      delta="not blocked or starved" progress={m.availability} />
          <MetricTile label="Performance" value={pct(m.performance, 1).replace("%", "")} unit="%"
                      delta="against 60/h takt" progress={m.performance} tone="starved" />
          <MetricTile label="Quality" value={pct(m.quality, 1).replace("%", "")} unit="%"
                      delta="first time through" progress={m.quality} tone="good" />
        </div>
      )}

      <div className="card p-5">
        <div className="text-lg font-medium">Where the constraint was</div>
        <div className="label mt-1 mb-4 leading-relaxed">
          Each band is a period with a different binding station. The bottleneck moves
          several times per shift &mdash; which is exactly what a static report cannot show.
        </div>
        <div className="flex h-9 rounded-md overflow-hidden border border-line">
          {data.bottleneck_runs.map((r, i) => (
            <div
              key={i}
              title={`${r.station} \u00b7 ${r.duration_min} min`}
              style={{
                width: `${(100 * (r.t_end - r.t_start)) / span}%`,
                // Hue from station index: stable and distinguishable with no
                // palette to maintain across 35 stations.
                background: `hsl(${(parseInt(r.station.slice(1)) * 47) % 360} 48% 62%)`,
              }}
              className="border-r border-white/50 font-mono text-[9px] text-white
                         flex items-center justify-center"
            >
              {r.duration_min > 22 ? r.station : ""}
            </div>
          ))}
        </div>
        <div className="label mt-3">
          {data.bottleneck_runs.length} distinct periods over {(span / 3600).toFixed(1)} hours
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="card p-5">
          <div className="text-lg font-medium mb-4">Disruptions this run</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-line">
                <th className="label font-normal pb-2">Station</th>
                <th className="label font-normal pb-2">Cause</th>
                <th className="label font-normal pb-2 text-right">Found in</th>
                <th className="label font-normal pb-2 text-right">Coverage</th>
              </tr>
            </thead>
            <tbody>
              {data.disruptions.map((d, i) => (
                <tr key={i} className="border-b border-line">
                  <td className="py-2 font-mono">{d.station}</td>
                  <td className="text-ink-muted">{d.cause}</td>
                  <td className="text-right font-mono">
                    {d.detected ? `${d.detect_lag_min} min`
                                : <span className="text-state-down">missed</span>}
                  </td>
                  <td className="text-right font-mono">{pct(d.hold_hit_rate, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card p-5">
          <div className="text-lg font-medium">Detection rate vs conventional methods</div>
          <div className="label mt-1 mb-4">
            Share of injected disruptions found &middot; 4 seeds &times; 10 disruptions
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={COMPARISON} layout="vertical" margin={{ left: 10, right: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} unit="%"
                       tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "monospace" }}
                       axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={140}
                       tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E5E7EB",
                                         borderRadius: 8, fontSize: 12 }}
                         formatter={(v) => [`${v}%`, "found"]} cursor={{ fill: "#F0F1F3" }} />
                <Bar dataKey="found" radius={[0, 3, 3, 0]} barSize={22}>
                  {COMPARISON.map((d, i) => (
                    <Cell key={i} fill={i === 0 ? "#8B2FE8" : "#D1D5DB"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
