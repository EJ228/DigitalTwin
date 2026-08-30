import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import EnginesView from "./views/EnginesView";
import EscapeWindowView from "./views/EscapeWindowView";
import LeadershipView from "./views/LeadershipView";
import ManagerView from "./views/ManagerView";
import SupervisorView from "./views/SupervisorView";
import { getLine, getRuns } from "./lib/api";
import { runName } from "./lib/format";
import { useIstClock } from "./lib/useIstClock";
import { useTwinStream } from "./lib/useTwinStream";

const TITLES = {
  supervisor: "Assembly Line 2",
  escape: "Escape window",
  manager: "Shift log",
  engines: "Model audit",
  leadership: "Rollout case",
};

export default function App() {
  const [tab, setTab] = useState("supervisor");
  const [run, setRun] = useState("run_s7");
  const [runs, setRuns] = useState([]);
  const [line, setLine] = useState(null);

  // Start shortly before drift onset so the demo reaches the alert quickly.
  const twin = useTwinStream({ run, initialT: 46000, speed: 60 });
  const ist = useIstClock();

  useEffect(() => {
    getLine().then(setLine).catch(() => {});
    getRuns()
      .then((r) => { setRuns(r); if (r.length && !r.find((x) => x.run === run)) setRun(r[0].run); })
      .catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const driftStation = line?.drift_station ?? "S08";
  const blindOn = twin.blind.includes(driftStation);
  const snap = twin.snapshot;

  // Sidebar andon list, derived from live state.
  const andons = [];
  if (snap?.drift?.alarm) {
    andons.push({ key: "drift", text: `ST ${driftStation.slice(1)} \u00b7 drift`,
                  shape: "rounded-[2px]", colour: "bg-alert" });
  }
  snap?.stations?.filter((s) => s.state === "down").slice(0, 2).forEach((s) =>
    andons.push({ key: s.id, text: `ST ${s.id.slice(1)} \u00b7 down`,
                  shape: "rounded-full", colour: "bg-state-down" }));
  if (snap?.bottleneck?.confident) {
    andons.push({ key: "btl", text: `ST ${snap.bottleneck.station.slice(1)} \u00b7 constraint`,
                  shape: "rotate-45", colour: "bg-state-bottleneck" });
  }

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-[1440px] mx-auto flex bg-page rounded-2xl overflow-hidden">
        <Sidebar tab={tab} setTab={setTab} andons={andons} />

        <div className="flex-1 min-w-0">
          <header className="bg-card px-7 py-4 flex items-center gap-x-6 gap-y-3 flex-wrap
                             border-b border-line rounded-tr-2xl">
            <div className="flex items-center gap-3 mr-auto">
              <h1 className="text-[22px] font-semibold">{TITLES[tab]}</h1>
              <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${
                twin.connected ? "bg-state-run/10" : "bg-state-down/10"}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${twin.connected ? "bg-state-run animate-pulse" : "bg-state-down"}`} />
                <span className={`font-mono text-[10px] tracking-label ${
                  twin.connected ? "text-state-run" : "text-state-down"}`}>
                  {twin.connected ? "LIVE" : "OFFLINE"}
                </span>
              </span>
            </div>

            {/* Replay controls, grouped so they read as one unit rather than
                four loose chips that wrap into a ragged block. */}
            <div className="flex items-center gap-1.5 p-1 rounded-lg bg-page">
              <select value={run} onChange={(e) => setRun(e.target.value)}
                      className="label bg-transparent rounded-md px-2 py-1.5 text-ink-muted
                                 hover:text-ink cursor-pointer focus:outline-none">
                {runs.map((r) => (
                  <option key={r.run} value={r.run}>{runName(r.run)}</option>
                ))}
              </select>

              <span className="w-px h-4 bg-line" />

              <button onClick={() => twin.setPaused(!twin.paused)}
                      className="label px-3 py-1.5 rounded-md text-ink-muted hover:bg-card
                                 hover:text-ink transition w-[62px]">
                {twin.paused ? "Play" : "Pause"}
              </button>

              <select defaultValue="60" onChange={(e) => twin.setSpeed(Number(e.target.value))}
                      className="label bg-transparent rounded-md px-2 py-1.5 text-ink-muted
                                 hover:text-ink cursor-pointer focus:outline-none">
                <option value="30">0.5&times;</option>
                <option value="60">1&times;</option>
                <option value="180">3&times;</option>
                <option value="600">10&times;</option>
              </select>
            </div>

            {/* The blind-station demo toggle. Kept separate from the replay
                controls because it changes the model, not the playback. */}
            <button
              onClick={() => twin.setBlind(blindOn ? "" : driftStation)}
              title={`Simulate ${driftStation} as a legacy bay with no instrumentation`}
              className={`label px-3 py-2 rounded-lg border transition ${
                blindOn ? "border-alert text-alert bg-alert/[0.06]"
                        : "border-line text-ink-muted hover:bg-page hover:text-ink"}`}
            >
              {blindOn ? `${driftStation} sensors off` : `Turn off ${driftStation} sensors`}
            </button>

            {/* Two clocks, and they are different things: the replay position in
                the recorded shift, and real IST wall time. Labelled so nobody has
                to guess which one they are reading. */}
            <div className="flex items-stretch gap-5 pl-5 border-l border-line">
              <div>
                <div className="label">Shift {snap?.shift ?? "\u2014"} &middot; takt 60s</div>
                <div className="figure text-lg mt-1 text-ink-muted">
                  {snap?.clock ?? "--:--"}
                  <span className="label ml-1.5 text-ink-faint">replay</span>
                </div>
              </div>
              <div className="text-right">
                <div className="label">India &middot; IST</div>
                <div className="figure text-lg mt-1">
                  {ist}
                  <span className="label ml-1.5 text-ink-faint">IST</span>
                </div>
              </div>
            </div>
          </header>

          {twin.error && (
            <div className="px-7 pt-4">
              <div className="rounded-lg border border-alert/40 bg-alert/[0.06] px-4 py-2.5 text-sm">
                {twin.error} &mdash; is the backend running?{" "}
                <span className="font-mono text-xs">make api</span>
              </div>
            </div>
          )}

          <main className="px-7 py-5">
            {tab === "supervisor" && (
              <SupervisorView snapshot={snap} run={run} blind={twin.blind}
                              driftStation={driftStation} />
            )}
            {tab === "manager" && <ManagerView snapshot={snap} run={run} blind={twin.blind} />}
            {tab === "leadership" && <LeadershipView run={run} blind={twin.blind} />}
            {tab === "escape" && <EscapeWindowView run={run} blind={twin.blind} />}
            {tab === "engines" && <EnginesView run={run} />}
          </main>
        </div>
      </div>
    </div>
  );
}
