import { useEffect, useState } from "react";
import { getEscapeWindow } from "../lib/api";

const TAKT_SECONDS = 60;

/**
 * THE MONEY SHOT.
 *
 * Two rows of the same cars in build order, one fault, two worlds. The
 * asymmetry has to read from across a room, so the bars are large, the flag is
 * a hard vertical rule, and nothing else competes for attention.
 */
export default function EscapeWindowView({ run, blind }) {
  const [d, setD] = useState(null);
  useEffect(() => { getEscapeWindow(run, blind).then(setD).catch(() => {}); }, [run, blind]);

  if (!d) return <div className="h-[420px] card animate-pulse" />;
  if (!d.available) {
    return (
      <div className="card p-8 text-sm text-ink-muted">
        This run has no injected drift, so there is no escape window to compare.
      </div>
    );
  }

  const saved = d.baseline_flag_at - d.twin_flag_at;
  const leadMin = Math.round((saved * TAKT_SECONDS) / 60);
  const total = Math.max(d.baseline_flag_at + 30, d.cars.length);

  return (
    <div className="space-y-5">
      <div className="card px-6 py-4 flex items-baseline justify-between flex-wrap gap-2">
        <div className="text-lg">
          Escape window &mdash; Station 08 fixture play,
          <span className="text-ink-muted"> one fault, two worlds</span>
        </div>
        <div className="label">1 square = 1 car &middot; build order left &rarr; right</div>
      </div>

      <Row
        kicker="Today" accent="alert"
        title="Reactive — the fault surfaces at end-of-line inspection"
        count={d.baseline_flag_at}
        flagAt={d.baseline_flag_at} total={total}
        note={`${d.baseline_flag_at} cars built with the fault \u2014 rework or scrap`}
        flagLabel={`INSPECTION FLAG \u00b7 CAR ${d.baseline_flag_at}`}
        cleanTone="neutral"
      />

      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-line" />
        <div className="label">Same line &middot; same fault &middot; same minute</div>
        <div className="flex-1 h-px bg-line" />
      </div>

      <Row
        kicker="With the twin" accent="accent"
        title="Drift caught mid-build — line corrected at the changeover"
        count={d.twin_flag_at}
        flagAt={d.twin_flag_at} total={total}
        note={`${d.twin_flag_at} affected`}
        flagLabel={`TWIN ALERT \u00b7 CAR ${d.twin_flag_at}`}
        cleanTone="good"
        cleanNote="The rest of the shift builds clean"
      />

      {/* Three summary cards. Each is label-on-top / figure-below so the numbers
          land on one baseline across the row; the first used to be a centred
          two-column split, which is why it sat visibly out of line with the
          other two. */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
        <div className="card px-6 py-5 flex flex-col">
          <div className="label leading-[1.5] min-h-[30px]">Escape window</div>
          <div className="flex items-baseline gap-3 mt-1">
            <span className="figure text-2xl text-ink-faint line-through">{d.baseline_flag_at}</span>
            <span className="text-ink-faint text-sm">&rarr;</span>
            <span className="figure text-[40px] text-accent">{d.twin_flag_at}</span>
            <span className="figure text-xl text-ink-muted ml-auto self-end pb-1">
              &minus;{d.reduction_pct}%
            </span>
          </div>
        </div>

        <div className="card px-6 py-5 flex flex-col">
          <div className="label leading-[1.5] min-h-[30px]">Cars saved &middot; per incident</div>
          <div className="figure text-[40px] text-state-run mt-1">{saved}</div>
        </div>

        <div className="card px-6 py-5 flex flex-col">
          <div className="label leading-[1.5] min-h-[30px]">Detection lead time</div>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className="figure text-[40px]">{leadMin}</span>
            <span className="text-lg text-ink-faint">min</span>
          </div>
        </div>
      </div>

      <div className="prose-note max-w-3xl">
        The gap is not inspection accuracy. It is the work in process between the drifting
        station and end-of-line inspection. Detecting at the station removes that lag entirely.
      </div>
    </div>
  );
}

function Row({ kicker, accent, title, count, flagAt, total, note, flagLabel, cleanTone, cleanNote }) {
  const border = accent === "alert" ? "border-alert" : "border-accent";
  const kickerColour = accent === "alert" ? "text-alert" : "text-accent";
  const countColour = accent === "alert" ? "text-alert" : "text-accent";
  const cleanBar = cleanTone === "good" ? "bg-state-run/30" : "bg-line";

  return (
    <div className={`card p-6 border-l-4 ${border}`}>
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <div className={`label ${kickerColour}`}>{kicker}</div>
          <div className="text-[19px] font-medium mt-2 leading-snug tracking-[-0.015em]">{title}</div>
        </div>
        <div className="flex items-baseline gap-2.5 shrink-0">
          <span className={`figure text-[44px] ${countColour}`}>{count}</span>
          <span className="label leading-[1.4] w-14">Cars<br />affected</span>
        </div>
      </div>

      <div className="mt-5 flex items-end gap-[2px] relative">
        {Array.from({ length: total }, (_, i) => {
          const affected = i < flagAt;
          return (
            <div key={i} className="relative flex-1 min-w-[3px]">
              {i === flagAt && (
                <div className={`absolute -top-1 bottom-0 -left-[2px] w-px
                                 ${accent === "alert" ? "bg-ink" : "bg-accent"}`} />
              )}
              <div
                className={`h-7 rounded-[1px] ${affected ? "bg-alert" : cleanBar}`}
                title={`car ${i}${affected ? " \u2014 built with the fault" : ""}`}
              />
            </div>
          );
        })}
      </div>

      <div className="mt-2 flex justify-between gap-6 flex-wrap">
        <div className={`label ${kickerColour}`}>{note}</div>
        <div className="flex gap-8">
          <div className="label text-ink-muted">{flagLabel}</div>
          {cleanNote && <div className="label text-state-run">{cleanNote}</div>}
        </div>
      </div>
    </div>
  );
}
