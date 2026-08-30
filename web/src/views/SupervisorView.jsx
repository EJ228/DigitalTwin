import { useEffect, useState } from "react";
import AlertBanner from "../components/AlertBanner";
import CounterfactualPanel from "../components/CounterfactualPanel";
import DriftChart from "../components/DriftChart";
import MetricTile from "../components/MetricTile";
import StationRibbon from "../components/StationRibbon";
import { getEscapeWindow, postCounterfactual } from "../lib/api";
import { num, pct, stateStyle } from "../lib/format";

/** Floor supervisor: what to do in the next thirty seconds. */
export default function SupervisorView({ snapshot, run, blind, driftStation }) {
  const [cf, setCf] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [acked, setAcked] = useState(false);
  const [ew, setEw] = useState(null);

  useEffect(() => { getEscapeWindow(run, blind).then(setEw).catch(() => {}); }, [run, blind]);
  useEffect(() => { setAcked(false); }, [run, blind]);

  if (!snapshot) return <Loading />;
  const { stations, bottleneck, drift, metrics } = snapshot;

  const investigate = async () => {
    setBusy(true);
    try { setCf(await postCounterfactual({ run, t: snapshot.t, blind, replicates: 5 })); }
    finally { setBusy(false); }
  };

  const sel = stations.find((s) => s.id === selected);
  const target = metrics.takt_target_per_hour;

  return (
    <div className="space-y-4">
      {!acked && (
        <AlertBanner
          drift={drift} station={driftStation} busy={busy}
          escapeAt={ew?.twin_flag_at}
          onInvestigate={investigate} onAck={() => setAcked(true)}
        />
      )}

      <div className="card p-5">
        <StationRibbon
          stations={stations} driftStation={driftStation}
          driftAlarm={drift?.alarm} onSelect={setSelected} selected={selected}
        />
        {sel && (
          <div className="mt-4 pt-3 border-t border-line font-mono text-xs text-ink-muted">
            {sel.id} &middot; {stateStyle(sel.state).label} &middot; buffer {sel.buffer}/{sel.buffer_capacity}
            {sel.blind && " \u00b7 no process sensors"}
            {sel.manual && " \u00b7 manual bay"}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_300px] gap-4 items-stretch">
        <div className="card p-5">
          <div className="text-[17px] font-medium leading-snug">
            Station {driftStation.slice(1)} &mdash; fixture play vs tolerance
          </div>
          <div className="label mt-1.5">Drift detector &middot; rolling 40 cars &middot; 1s refresh</div>
          <div className="mt-4">
            <DriftChart drift={drift} station={driftStation} />
          </div>
          <div className="prose-note mt-3 pt-3 border-t border-line">
            Band shown at 90%. The twin reports its own uncertainty rather than smoothing it away.
          </div>
        </div>

        {/* Flex column so the four tiles divide the chart card's height
            exactly. With space-y-4 the stack ran to its own natural height
            and finished 42px below the chart, leaving the two columns
            visibly out of line at the bottom. */}
        <div className="flex flex-col gap-4">
          <MetricTile
            label="Throughput &middot; cars / hr"
            value={num(metrics.throughput_per_hour, 0)}
            denom={num(target, 0)}
            delta={`${metrics.throughput_per_hour >= target ? "\u25B2" : "\u25BC"} ${num(Math.abs(metrics.throughput_per_hour - target), 0)} vs target`}
            progress={metrics.throughput_per_hour / target}
            tone={metrics.throughput_per_hour >= target ? "good" : "alert"}
          />
          <MetricTile
            label="Cars built &middot; this shift"
            value={metrics.built_total}
            denom="480"
            progress={metrics.built_total / 480}
            tone="starved"
          />
          <MetricTile
            label="First time through"
            value={pct(metrics.first_time_through, 1).replace("%", "")}
            unit="%"
            progress={metrics.first_time_through}
            tone={metrics.first_time_through > 0.95 ? "good" : "alert"}
          />
          <MetricTile
            label="Constraint"
            value={bottleneck.confident ? bottleneck.station : "\u2014"}
            delta={bottleneck.confident ? `ahead of ${bottleneck.runner_up}` : "line has slack"}
            tone={bottleneck.confident ? "accent" : "ink"}
          />
        </div>
      </div>

      {cf && <CounterfactualPanel result={cf} onClose={() => setCf(null)} />}
    </div>
  );
}

function Loading() {
  return (
    <div className="space-y-4">
      <div className="h-[104px] card animate-pulse" />
      <div className="h-[132px] card animate-pulse" />
      <div className="h-[360px] card animate-pulse" />
    </div>
  );
}
