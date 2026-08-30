import {
  Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

/**
 * The single most important chart in the product.
 *
 * Observed drift climbs toward the tolerance threshold; where it crosses, the
 * alert fires. Past that point the dashed line is an LSTM forecast and the
 * shaded band is a split-conformal residual interval -- BOTH COMPUTED BY THE
 * BACKEND. An earlier version of this component drew the band itself from two
 * constants, which made a shape invented in the frontend look exactly like a
 * model output. Nothing here fabricates numbers any more: if the backend sends
 * no projection, none is drawn.
 *
 * When the station has no sensors we do not draw an empty chart. We show the
 * Gaussian-process posterior instead and say what we cannot see. A twin that
 * hides its own ignorance is worse than no twin.
 */
export default function DriftChart({ drift, station }) {
  if (!drift?.available) return <BlindPanel drift={drift} station={station} />;

  const obs = drift.series ?? [];
  const proj = drift.projection ?? [];
  const th = drift.threshold;
  const alarmT = drift.alarm_t;

  const data = [
    ...obs.map((p) => ({ t: p.t, observed: p.v })),
    ...proj.map((p) => ({ t: p.t, projected: p.mu, band: [p.lo, p.hi] })),
  ].sort((a, b) => a.t - b.t);

  // join the observed and projected lines so the chart reads as one trajectory
  if (proj.length && obs.length) {
    const last = data.findIndex((d) => d.projected != null);
    if (last > 0) data[last - 1].projected = data[last - 1].observed;
  }

  return (
    <div>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 20, right: 24, bottom: 8, left: 8 }}>
            <XAxis dataKey="t" tick={false} axisLine={{ stroke: "#E5E7EB" }} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'JetBrains Mono Variable', monospace" }}
                   axisLine={false} tickLine={false} width={44} />
            <Tooltip
              contentStyle={{ background: "#fff", border: "1px solid #E5E7EB",
                              borderRadius: 8, fontSize: 12 }}
              formatter={(v, n) => (v == null ? null : [Number(v).toFixed(2), n])}
              labelFormatter={() => ""}
            />
            <ReferenceLine
              y={th} stroke="#D2551E" strokeDasharray="5 4"
              label={{ value: "TOLERANCE THRESHOLD", fill: "#D2551E", fontSize: 9,
                       fontFamily: "monospace", position: "insideBottomLeft", offset: 8 }}
            />
            {alarmT != null && (
              <ReferenceLine x={alarmT} stroke="#D2551E" strokeDasharray="3 3"
                label={{ value: "ALERT FIRED", fill: "#D2551E", fontSize: 9,
                         fontFamily: "monospace", position: "top" }} />
            )}
            {proj.length > 0 && (
              <Area dataKey="band" stroke="none" fill="#8B2FE8" fillOpacity={0.12}
                    isAnimationActive={false} connectNulls />
            )}
            <Line dataKey="observed" stroke="#D2551E" strokeWidth={1.8} dot={false}
                  isAnimationActive={false} connectNulls={false} name="observed" />
            {proj.length > 0 && (
              <Line dataKey="projected" stroke="#8B2FE8" strokeWidth={1.6} dot={false}
                    strokeDasharray="5 4" isAnimationActive={false} connectNulls={false}
                    name="forecast" />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="prose-note mt-3 pt-3 border-t border-line">
        {proj.length > 0 && drift.projection_band ? (
          <>
            Forecast band at {Math.round(100 * drift.projection_band.level)}% &middot;{" "}
            {drift.projection_band.source} &middot; half-width{" "}
            {drift.projection_band.half_width}
          </>
        ) : (
          <>No forecast available for this window &mdash; observed series only.</>
        )}
        {drift.false_alarm_budget_parts && (
          <> &middot; alert budget 1 false alarm per {drift.false_alarm_budget_parts} parts</>
        )}
      </div>
    </div>
  );
}

function BlindPanel({ drift, station }) {
  const bc = drift?.blind_confidence;
  return (
    <div className="h-[300px] rounded-lg border border-dashed border-line
                    flex flex-col items-center justify-center text-center px-8">
      <div className="label text-ink-muted">No process sensors at {station}</div>
      <div className="text-sm text-ink-muted mt-3 max-w-md leading-relaxed">
        {drift?.reason}
      </div>

      {bc?.blinded && bc.posterior_sd_seconds != null ? (
        <div className="mt-5 bg-page rounded-lg px-5 py-4 max-w-md">
          <div className="label">Inferred from {bc.neighbours?.length ?? 0} neighbouring stations</div>
          <div className="flex items-baseline justify-center gap-2 mt-2">
            <span className="font-mono text-3xl text-accent">
              &plusmn;{bc.posterior_sd_seconds}s
            </span>
            <span className="label">posterior sd</span>
          </div>
          <div className="prose-note mt-2">
            {bc.source}. Without a model the uncertainty would be
            &plusmn;{bc.naive_sd_seconds}s. Bottleneck detection continues unaffected,
            because it only ever needed timestamps.
          </div>
        </div>
      ) : (
        <div className="text-xs text-ink-faint mt-4 max-w-md leading-relaxed">
          Bottleneck detection continues from entry and exit timestamps alone. The
          station is marked blind rather than silently assumed healthy.
        </div>
      )}
    </div>
  );
}
