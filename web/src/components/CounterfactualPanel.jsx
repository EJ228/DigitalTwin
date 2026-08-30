import { num } from "../lib/format";

/**
 * Monte Carlo rollout results.
 *
 * Deltas are against "do nothing", because that is the real alternative. The
 * spread across replicates sits next to every mean, so a difference smaller
 * than the noise looks smaller than the noise.
 */
export default function CounterfactualPanel({ result, onClose }) {
  if (!result) return null;
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-base font-medium">If we act now &mdash; {result.clock}</div>
          <div className="label mt-1">
            {result.replicates} simulated futures per option &middot; {result.horizon_min} minute horizon
          </div>
        </div>
        <button onClick={onClose} className="label hover:text-ink">close</button>
      </div>

      {/* Column sizing: the option column takes w-full and therefore absorbs all
          slack, which squeezes the four numeric columns down to their own
          content. Each is nowrap, so a column is always exactly as wide as its
          widest cell and the separation comes from padding.
          This replaces both earlier attempts. Auto layout alone let the numbers
          collide into one run of digits ("52.6 63.2 \u00b12.4 11.2 \u2014"); pinning the
          columns to fixed px then broke the other way, because a browser
          minimum-font-size setting scales the 10px headers up while a px width
          stays put, wrapping "Good units/h" over two lines and overlapping its
          neighbours. Content-sized columns hold at any text size. */}
      <table className="w-full mt-4 text-sm">
        <thead>
          <tr className="text-left border-b border-line align-bottom">
            <th className="label font-normal pb-2.5 w-full">Option</th>
            <th className="label font-normal pb-2.5 text-right pl-6 whitespace-nowrap">Good units/h</th>
            <th className="label font-normal pb-2.5 text-right pl-6 whitespace-nowrap">Throughput</th>
            <th className="label font-normal pb-2.5 text-right pl-6 whitespace-nowrap">Defects/h</th>
            <th className="label font-normal pb-2.5 text-right pl-6 whitespace-nowrap">Downtime</th>
          </tr>
        </thead>
        <tbody>
          {result.candidates.map((c) => {
            const best = c.key === result.recommended;
            return (
              <tr key={c.key} className={`border-b border-line ${best ? "bg-accent/[0.05]" : ""}`}>
                <td className="py-3.5 pr-6">
                  <div className={best ? "text-accent font-medium" : "font-medium"}>{c.label}</div>
                  <div className="text-xs text-ink-muted leading-snug mt-1">{c.detail}</div>
                </td>
                <td className="text-right font-mono align-top py-3.5 pl-6 whitespace-nowrap">
                  {num(c.good_units_per_hour)}
                </td>
                <td className="text-right font-mono align-top py-3.5 pl-6 whitespace-nowrap">
                  {num(c.throughput_per_hour)}
                  <span className="text-ink-faint ml-1.5">&plusmn;{num(c.throughput_sd, 1)}</span>
                </td>
                <td className="text-right font-mono align-top py-3.5 pl-6 whitespace-nowrap">
                  {num(c.defects_per_hour)}
                </td>
                <td className="text-right font-mono align-top py-3.5 pl-6 whitespace-nowrap text-ink-muted">
                  {c.downtime_min ? `${c.downtime_min} min` : "\u2014"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {result.payback_note && (
        <div className="mt-4 text-sm bg-page rounded-lg px-4 py-3">{result.payback_note}</div>
      )}
      <div className="prose-note mt-3 text-ink-faint">
        {result.caveat}
      </div>
    </div>
  );
}
