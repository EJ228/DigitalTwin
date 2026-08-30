/**
 * The Andon banner.
 *
 * One line of what, one of why it was invisible until now, one action. A
 * supervisor reads this standing up, mid-task, in noise -- so the station number
 * is the largest thing in it and the recommended action sits next to the button
 * that acts on it.
 *
 * Layout is an explicit grid rather than a wrapping flex row. Wrapping put the
 * buttons underneath the station number with nothing to align to, which is what
 * made the banner look broken at narrow widths; the grid keeps the four regions
 * on one baseline and collapses them in a defined order instead.
 *
 * Every figure shown here comes from the backend. An earlier version displayed a
 * literal "Confidence 92%" that nothing computed; it has been replaced by the
 * statistic's exceedance over its calibrated threshold and the false-alarm
 * budget that threshold was set to, both of which are real.
 */
export default function AlertBanner({ drift, station, onInvestigate, onAck, busy, escapeAt }) {
  if (!drift?.alarm) return null;
  const ex = drift.exceedance;
  const budget = drift.false_alarm_budget_parts;

  return (
    <div className="card shadow-raised overflow-hidden flex items-stretch border-l-[3px] border-alert">
      {/* Rail: fixed width, vertically centred, never participates in wrapping. */}
      <div className="w-[74px] shrink-0 bg-alert/[0.06] flex flex-col items-center justify-center gap-2">
        <span className="w-2 h-2 bg-alert rounded-[2px]" />
        <span className="label text-alert">Andon</span>
      </div>

      <div
        className="flex-1 min-w-0 px-6 py-5 grid items-center gap-x-7 gap-y-5
                   grid-cols-1
                   md:grid-cols-[auto_minmax(0,1fr)]
                   xl:grid-cols-[auto_minmax(0,1fr)_auto_auto]"
      >
        {/* Station number */}
        <div className="shrink-0">
          <div className="label text-alert/70">Station</div>
          <div className="figure text-alert text-[40px] mt-1">{station.slice(1)}</div>
        </div>

        {/* What and why */}
        <div className="min-w-0">
          <div className="text-[19px] font-medium leading-snug tracking-[-0.015em]">
            Fixture play — progressive drift
          </div>
          <div className="prose-note mt-1.5">
            No single sensor is outside tolerance. This exists only in the
            relationship between two gaps
            {escapeAt != null && <>, caught at car {escapeAt}</>}.
          </div>
          <div className="label mt-2 text-ink-muted">
            {ex != null && <>{ex}× threshold</>}
            {ex != null && budget && <span className="text-ink-faint"> · </span>}
            {budget && <>1 false alarm / {budget} parts</>}
          </div>
        </div>

        {/* Recommended action */}
        <div className="xl:border-l xl:border-line xl:pl-7 min-w-0 xl:max-w-[220px]">
          <div className="label">Recommended action</div>
          <div className="text-[15px] mt-1.5 leading-snug">
            Re-torque fixture at next changeover
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 shrink-0">
          <button onClick={onAck}
            className="px-5 py-2.5 rounded-lg bg-alert text-white text-[10px] font-mono
                       uppercase tracking-label hover:brightness-95 transition">
            Ack
          </button>
          <button onClick={onInvestigate} disabled={busy}
            className="px-5 py-2.5 rounded-lg border border-line text-[10px] font-mono
                       uppercase tracking-label text-ink-muted hover:bg-page
                       disabled:opacity-50 transition">
            {busy ? "Simulating" : "Details"}
          </button>
        </div>
      </div>
    </div>
  );
}
