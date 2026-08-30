/**
 * The right-hand metric card: micro-label, oversized figure, secondary
 * denominator, progress rule. Deliberately readable from two metres away.
 *
 * The label and the delta used to share one flex row, so a two-word label wrapped
 * underneath the delta and every tile ended up a different height with the
 * figures on different baselines. The label now owns its row at a fixed two-line
 * height and the delta sits with the figure it qualifies, which keeps the big
 * numbers aligned across the stack no matter how long the labels are.
 */
export default function MetricTile({ label, value, denom, unit, delta, progress, tone = "ink" }) {
  const tones = {
    ink: "text-ink",
    alert: "text-alert",
    accent: "text-accent",
    good: "text-state-run",
    starved: "text-state-starved",
  };
  const bars = {
    ink: "bg-ink",
    alert: "bg-alert",
    accent: "bg-accent",
    good: "bg-state-run",
    starved: "bg-state-starved",
  };
  return (
    <div className="card px-5 py-4 flex flex-col flex-1 min-h-[124px]">
      {/* Two-line box: short and long labels occupy the same vertical space. */}
      <div className="label leading-[1.5] min-h-[30px]">{label}</div>

      <div className="flex items-baseline flex-wrap gap-x-1.5 gap-y-1 mt-1">
        <span className={`figure text-[40px] ${tones[tone]}`}>{value}</span>
        {unit && <span className={`text-base ${tones[tone]}`}>{unit}</span>}
        {denom && <span className="text-lg text-ink-faint">/ {denom}</span>}
        {delta && (
          <span className="label text-ink-muted ml-auto self-end pb-1 whitespace-nowrap">
            {delta}
          </span>
        )}
      </div>

      {/* Reserve the rule's space even when absent, and pin it to the bottom with
          mt-auto. The tiles share the column height, so any slack collects above
          the rule and all four rules sit on the same line as each other. */}
      <div className="mt-auto pt-4">
        <div className="h-[3px] rounded-full overflow-hidden bg-line">
          {progress != null && (
            <div
              className={`h-full rounded-full ${bars[tone]} transition-[width] duration-500`}
              style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
