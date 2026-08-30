import { useEffect, useState } from "react";
import {
  Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { getCoherence, getEngines, getEnginesAvailability } from "../lib/api";
import { featureName, num, pct, runName } from "../lib/format";

/**
 * Model audit: the four engines that are not the live line.
 *
 * Every figure here comes from /api/engines, which serves the same evaluation
 * that writes results/RESULTS.md. The dashboard cannot show a number the
 * pipeline did not produce.
 */
export default function EnginesView({ run }) {
  const [e, setE] = useState(null);
  const [coh, setCoh] = useState(null);
  const [err, setErr] = useState(null);
  const [avail, setAvail] = useState(null);
  const [running, setRunning] = useState(false);

  // Readiness first. Fetching the engines outright would block for minutes on a
  // run with no completed offline evaluation, and a tab click should never
  // start work that long without being asked for.
  useEffect(() => {
    setE(null); setErr(null); setAvail(null); setRunning(false);
    getCoherence(run).then(setCoh).catch(() => {});
    getEnginesAvailability(run)
      .then((a) => {
        setAvail(a);
        if (a.precomputed || a.cached) {
          setRunning(true);
          getEngines(run).then(setE).catch(() => setErr("engine evaluation failed"));
        }
      })
      .catch(() => setErr("could not reach the evaluation service"));
  }, [run]);

  const evaluateNow = () => {
    setRunning(true);
    getEngines(run).then(setE).catch(() => setErr("engine evaluation failed"));
  };

  if (err) return <div className="card p-6 text-sm text-alert">{err}</div>;

  // Ready to serve, or already running: show the skeleton.
  if (!e && running) {
    const quick = avail?.precomputed || avail?.cached;
    return (
      <div className="space-y-4">
        <div className="card p-6">
          <div className="text-[15px] font-medium">
            {quick ? "Loading the evaluation…" : `Evaluating ${runName(run)}…`}
          </div>
          {!quick && (
            <div className="prose-note mt-2 max-w-2xl">
              Training the hazard model, a buffer forecaster per station and GraphSAGE.
              This takes several minutes. The result is cached, so coming back to
              {" "}{runName(run)} afterwards is instant.
            </div>
          )}
        </div>
        <div className="h-64 card animate-pulse" />
      </div>
    );
  }

  // Not evaluated, and evaluating is expensive: offer it instead of doing it.
  if (!e) {
    if (!avail) return <div className="h-64 card animate-pulse" />;
    return (
      <div className="card p-6">
        <div className="text-[17px] font-medium">
          No completed evaluation for {runName(run)}
        </div>
        <div className="prose-note mt-2 max-w-2xl">
          The offline pipeline evaluates one run, and this is not it. Every figure on
          this tab is measured, never carried over from another run, so there is
          nothing to show until {runName(run)} has actually been evaluated. That means
          training the hazard model, a buffer forecaster per station and GraphSAGE —
          several minutes of compute.
        </div>
        <button
          onClick={evaluateNow}
          className="label mt-4 px-4 py-2.5 rounded-lg bg-accent text-white
                     hover:brightness-95 transition"
        >
          Evaluate {runName(run)} now
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Provenance run={run} computed={e.computed} testedOn={e.hazard?.tested_on} />
      <HazardCard h={e.hazard} />
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <ForecastCard f={e.forecast} />
        <GraphSageCard g={e.graphsage} />
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <BlindCard b={e.blind} />
        <div className="space-y-4">
          <CoherenceCard c={e.coherence} series={coh} />
          <ConformalCard k={e.conformal} />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

/**
 * Which run these numbers describe, and where they came from.
 *
 * The tab is driven by the run selector, but the underlying evaluation is
 * expensive and only one run is precomputed. Stating the run and the provenance
 * on the page means the figures can never be read as belonging to a run they
 * were not measured on.
 */
function Provenance({ run, computed, testedOn }) {
  if (!testedOn) return null;
  const stale = testedOn !== run;
  return (
    <div className={`card px-5 py-3 flex items-center justify-between gap-4 flex-wrap ${
      stale ? "border-l-[3px] border-alert" : ""}`}>
      <div className="prose-note">
        Evaluated on <span className="text-ink font-medium">{runName(testedOn)}</span>
        {stale && (
          <> — not the selected run ({runName(run)}). These figures describe {runName(testedOn)}.</>
        )}
      </div>
      <div className={`label ${computed === "precomputed" ? "text-ink-faint" : "text-state-run"}`}>
        {computed === "precomputed" ? "Offline evaluation" : "Computed on demand"}
      </div>
    </div>
  );
}

function Head({ title, sub, badge, tone = "accent" }) {
  const tones = { accent: "text-accent", good: "text-state-run", alert: "text-alert" };
  return (
    <div className="flex items-start justify-between gap-6">
      <div className="min-w-0">
        <div className="text-[17px] font-medium leading-snug">{title}</div>
        {sub && <div className="prose-note mt-2 max-w-2xl">{sub}</div>}
      </div>
      {badge && <div className={`label shrink-0 ${tones[tone]}`}>{badge}</div>}
    </div>
  );
}

function Stat({ label, value, sub, tone = "ink" }) {
  const tones = { ink: "text-ink", accent: "text-accent", good: "text-state-run",
                  alert: "text-alert", faint: "text-ink-faint" };
  return (
    <div className="min-w-0">
      <div className="label leading-[1.5] min-h-[30px]">{label}</div>
      <div className={`figure text-[28px] ${tones[tone]}`}>{value}</div>
      {sub && <div className="prose-note mt-1.5 text-[11.5px]">{sub}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function HazardCard({ h }) {
  if (!h) return null;
  const max = Math.max(...h.top_features.map((f) => f.importance)) || 1;
  return (
    <div className="card p-5">
      <Head
        title={`Hazard model \u2014 defect risk at ${h.station} from upstream stations only`}
        sub={`Trained on ${h.trained_on.map(runName).join(", ")}, tested on ${runName(h.tested_on)}. Every downstream column is dropped before training, so the model cannot explain a defect after the fact \u2014 it has to predict. Labels are end-of-line inspection outcomes, never cause attribution.`}
        badge={h.leak_check.clean ? "\u2713 NO DOWNSTREAM LEAK" : "LEAK DETECTED"}
        tone={h.leak_check.clean ? "good" : "alert"}
      />

      <div className="grid grid-cols-2 md:grid-cols-5 gap-6 mt-5 pt-5 border-t border-line">
        <Stat label="MCC" value={num(h.mcc, 3)} tone="accent"
              sub="Matthews correlation" />
        <Stat label="ROC AUC" value={num(h.auc, 3)} />
        <Stat label="Avg precision" value={num(h.average_precision, 3)}
              sub={`base rate ${pct(h.positive_rate)}`} />
        <Stat label="Predict-all-good MCC" value={num(h.baselines.predict_all_good_mcc, 3)}
              tone="faint" sub={`but ${pct(h.baselines.predict_all_good_accuracy)} accurate`} />
        <Stat label="Features" value={h.n_features} tone="faint"
              sub={`${h.n_train} train / ${h.n_test} test`} />
      </div>

      <div className="mt-5 pt-5 border-t border-line">
        <div className="label mb-3">
          Attribution &middot; {h.top_features[0]?.method === "shap" ? "SHAP" : "permutation"} importance
        </div>
        <div className="space-y-1.5">
          {h.top_features.slice(0, 6).map((f) => (
            <div key={f.feature} className="flex items-center gap-3">
              <div className="text-[13px] w-56 shrink-0 truncate text-ink-muted" title={f.feature}>
                {featureName(f.feature)}
              </div>
              <div className="flex-1 h-2 bg-page rounded-full overflow-hidden">
                <div className="h-full bg-accent rounded-full"
                     style={{ width: `${(100 * f.importance) / max}%` }} />
              </div>
            </div>
          ))}
        </div>
        <div className="prose-note mt-3">
          A model predicting &ldquo;fine&rdquo; for every unit scores{" "}
          {pct(h.baselines.predict_all_good_accuracy)} accuracy and an MCC of zero. That is
          why accuracy is not reported as a headline here.
        </div>
      </div>
    </div>
  );
}

function ForecastCard({ f }) {
  if (!f) return null;
  const won = f.beats_persistence_at;
  return (
    <div className="card p-5">
      <Head
        title="LSTM buffer forecaster"
        sub={`Forecasts buffer fill across all 35 stations from a ${f.lookback_min}-minute lookback, so the constraint can be predicted rather than reported. Hand-written in NumPy with BPTT and Adam \u2014 no framework.`}
        badge={`BEATS PERSISTENCE ${won}`}
        tone={won.split("/")[0] === won.split("/")[1] ? "good" : "alert"}
      />
      <table className="w-full mt-4 text-sm">
        <thead>
          <tr className="text-left border-b border-line">
            <th className="label font-normal pb-2">Horizon</th>
            <th className="label font-normal pb-2 text-right">LSTM</th>
            <th className="label font-normal pb-2 text-right">Persistence</th>
            <th className="label font-normal pb-2 text-right">Linear</th>
            <th className="label font-normal pb-2 text-right">Skill</th>
          </tr>
        </thead>
        <tbody>
          {f.horizons.map((r) => (
            <tr key={r.horizon_min} className="border-b border-line">
              <td className="py-2 font-mono">{r.horizon_min} min</td>
              <td className="text-right font-mono text-accent">{r.lstm_rmse.toFixed(4)}</td>
              <td className="text-right font-mono text-ink-muted">{r.persistence_rmse.toFixed(4)}</td>
              <td className="text-right font-mono text-ink-faint">{r.linear_rmse.toFixed(4)}</td>
              <td className={`text-right font-mono ${r.beats_persistence ? "text-state-run" : "text-alert"}`}>
                {r.skill_vs_persistence >= 0 ? "+" : ""}{r.skill_vs_persistence.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="prose-note mt-3">{f.note}</div>
    </div>
  );
}

function GraphSageCard({ g }) {
  if (!g) return null;
  const a = g.results.graphsage, b = g.results.no_aggregation;
  return (
    <div className="card p-5">
      <Head
        title="GraphSAGE on the station topology"
        sub={`Two-layer mean aggregator (Hamilton et al. 2017), NumPy. Task: ${g.task}. Labels are the detector\u2019s own output shifted forward, so it is self-supervised on the event log.`}
        badge={g.aggregation_helps ? "AGGREGATION HELPS" : "NO GAIN FROM GRAPH"}
        tone={g.aggregation_helps ? "good" : "alert"}
      />
      <table className="w-full mt-4 text-sm">
        <thead>
          <tr className="text-left border-b border-line">
            <th className="label font-normal pb-2">Model</th>
            <th className="label font-normal pb-2 text-right">ROC AUC</th>
            <th className="label font-normal pb-2 text-right">MCC</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-line">
            <td className="py-2 text-accent font-medium">GraphSAGE (mean aggregator)</td>
            <td className="text-right font-mono text-accent">{a.auc.toFixed(3)}</td>
            <td className="text-right font-mono text-accent">{a.mcc.toFixed(3)}</td>
          </tr>
          <tr className="border-b border-line">
            <td className="py-2 text-ink-muted">Same net, aggregation off</td>
            <td className="text-right font-mono text-ink-muted">{b.auc.toFixed(3)}</td>
            <td className="text-right font-mono text-ink-muted">{b.mcc.toFixed(3)}</td>
          </tr>
          <tr>
            <td className="py-2 label">Difference</td>
            <td className={`text-right font-mono ${g.delta_auc > 0 ? "text-state-run" : "text-alert"}`}>
              {g.delta_auc >= 0 ? "+" : ""}{g.delta_auc.toFixed(3)}
            </td>
            <td className={`text-right font-mono ${g.delta_mcc > 0 ? "text-state-run" : "text-alert"}`}>
              {g.delta_mcc >= 0 ? "+" : ""}{g.delta_mcc.toFixed(3)}
            </td>
          </tr>
        </tbody>
      </table>
      <div className="prose-note mt-3">{g.note}</div>
    </div>
  );
}

function BlindCard({ b }) {
  if (!b) return null;
  return (
    <div className="card p-5">
      <Head
        title="Blind-station engine"
        sub="Virtual metrology infers a station from its neighbours; a Gaussian process says how sure we are; Expected Information Gain ranks where a sensor buys the most."
        badge={`SKILL POSITIVE AT ${b.n_with_positive_skill}/${b.n_evaluated}`}
        tone={b.n_with_positive_skill > b.n_evaluated / 2 ? "good" : "alert"}
      />

      <div className="prose-note mt-4 bg-page rounded-lg p-3">
        {b.validation}
      </div>

      <table className="w-full mt-4 text-sm">
        <thead>
          <tr className="text-left border-b border-line">
            <th className="label font-normal pb-2">Station</th>
            <th className="label font-normal pb-2">Sensing</th>
            <th className="label font-normal pb-2 text-right">Skill</th>
            <th className="label font-normal pb-2 text-right">Posterior sd</th>
            <th className="label font-normal pb-2 text-right">EIG bits</th>
          </tr>
        </thead>
        <tbody>
          {b.ranking.slice(0, 8).map((r) => (
            <tr key={r.station} className="border-b border-line">
              <td className="py-2 font-mono">{r.station}</td>
              <td className={r.tier === "blind" ? "text-alert" : "text-ink-muted"}>
                {r.tier === "blind" ? "none" : r.tier}
              </td>
              <td className={`text-right font-mono ${r.inference_skill > 0 ? "text-state-run" : "text-ink-faint"}`}>
                {r.inference_skill == null ? "\u2014" : r.inference_skill.toFixed(2)}
              </td>
              <td className="text-right font-mono">{r.posterior_sd ?? "\u2014"}</td>
              <td className="text-right font-mono text-accent">{r.eig_bits}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="prose-note mt-3">
        Skill is 1 &minus; rmse/naive, where naive is the station&rsquo;s own historical mean.
        Zero means neighbours told us nothing beyond what we already knew &mdash; reported
        rather than tuned away.
      </div>
    </div>
  );
}

function CoherenceCard({ c, series }) {
  if (!c?.available) return null;
  const data = (series?.series ?? []).map((r) => ({
    t: r.t, observed: r.wip_observed, predicted: r.wip_predicted,
  }));
  return (
    <div className="card p-5">
      <Head
        title="Little's Law self-audit"
        sub="WIP = throughput × flow time. An identity, not a model — so when the twin stops satisfying it, the twin has drifted from the line."
        badge={c.coherent ? "\u2713 COHERENT" : "DRIFTED"}
        tone={c.coherent ? "good" : "alert"}
      />
      <div className="flex gap-8 mt-4">
        <Stat label="Mean WIP error" value={pct(c.mean_abs_error)}
              tone={c.coherent ? "good" : "alert"}
              sub={`tolerance ${pct(c.tolerance, 0)}`} />
        <Stat label="Windows breaching" value={`${c.n_breaches}/${c.n_windows}`} tone="faint" />
      </div>
      {data.length > 0 && (
        <div className="h-36 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
              <XAxis dataKey="t" tick={false} axisLine={{ stroke: "#E5E7EB" }} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#9CA3AF", fontFamily: "'JetBrains Mono Variable', monospace" }}
                     axisLine={false} tickLine={false} width={34} />
              {/* Round in the tooltip. Recharts prints the raw float otherwise,
                  which surfaced values like 77.45667370247702 on hover. */}
              <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E5E7EB",
                                       borderRadius: 8, fontSize: 12 }}
                       formatter={(v, n) => (v == null ? null : [Number(v).toFixed(1), n])}
                       labelFormatter={() => ""} />
              <Line dataKey="observed" stroke="#1A1A1F" strokeWidth={1.6} dot={false}
                    isAnimationActive={false} name="WIP observed" />
              <Line dataKey="predicted" stroke="#8B2FE8" strokeWidth={1.4} dot={false}
                    strokeDasharray="4 3" isAnimationActive={false} name="Little's Law" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="prose-note mt-2">{c.verdict}</div>
    </div>
  );
}

function ConformalCard({ k }) {
  if (!k) return null;
  return (
    <div className="card p-5">
      <Head
        title="Conformal alert calibration"
        sub="Distribution-free and finite-sample: no normality, no asymptotics, only exchangeability of the in-control scores."
        badge={k.finite_sample_exact ? "\u2713 EXACT" : "CONSERVATIVE"}
        tone={k.finite_sample_exact ? "good" : "alert"}
      />
      <div className="grid grid-cols-2 gap-6 mt-4">
        <Stat label="Threshold" value={num(k.threshold, 2)} tone="accent"
              sub={`from ${k.calibration_n} in-control scores`} />
        <Stat label="Escape window" value={k.escape_window_parts ?? "\u2014"}
              sub={`parts \u00b7 ${k.runs_detected} runs detected`} />
        <Stat label="Nominal false alarms" value={`1 / ${Math.round(1 / k.alpha)}`}
              tone="faint" sub="parts" />
        <Stat label="Empirical, held out"
              value={k.empirical_arl0_parts ? `1 / ${Math.round(k.empirical_arl0_parts)}`
                                            : "0 alarms"}
              tone="faint" sub="drift-free runs" />
      </div>
      <div className="prose-note mt-4 bg-page rounded-lg p-3">
        {k.guarantee}
      </div>
    </div>
  );
}
