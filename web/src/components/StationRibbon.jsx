import { stateStyle } from "../lib/format";

const ZONES = [
  { key: "body", label: "BODY", range: "01-12" },
  { key: "paint", label: "PAINT", range: "13-22" },
  { key: "final", label: "FINAL ASSEMBLY", range: "23-35" },
];

/**
 * All 35 stations in build order, grouped by zone.
 *
 * Markers sit ABOVE the cells rather than inside them, so the two things a
 * supervisor scans for -- where the constraint is and where the drift is --
 * are visible without reading any cell contents.
 */
export default function StationRibbon({ stations, driftStation, driftAlarm, onSelect, selected }) {
  if (!stations?.length) {
    return <div className="h-24 rounded-lg bg-page animate-pulse" />;
  }
  return (
    <div>
      <div className="flex items-start justify-between mb-3">
        <div className="label">Station ribbon &middot; 35 stations in build order</div>
        <Legend />
      </div>

      <div className="flex gap-8 flex-wrap">
        {ZONES.map((z) => (
          <div key={z.key}>
            <div className="flex gap-1">
              {stations.filter((s) => s.zone === z.key).map((s) => (
                <Cell
                  key={s.id}
                  s={s}
                  drift={s.id === driftStation && driftAlarm}
                  selected={selected === s.id}
                  onSelect={onSelect}
                />
              ))}
            </div>
            <div className="label mt-2">{z.label} &middot; {z.range}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Cell({ s, drift, selected, onSelect }) {
  const st = stateStyle(s.state);
  const code = drift ? "DRF" : s.is_bottleneck ? "BTL" : st.code;

  const border = drift
    ? "border-alert"
    : s.is_bottleneck
    ? "border-state-bottleneck"
    : s.state === "down"
    ? "border-state-down/50"
    : "border-line";
  const bg = drift
    ? "bg-alert/[0.06]"
    : s.is_bottleneck
    ? "bg-state-bottleneck/[0.06]"
    : s.state === "down"
    ? "bg-state-down/[0.05]"
    : "bg-card";
  const codeColour = drift
    ? "text-alert"
    : s.is_bottleneck
    ? "text-state-bottleneck"
    : st.text;

  return (
    <div className="relative">
      {/* markers above the ribbon */}
      <div className="h-4 flex items-end justify-center">
        {drift && <span className="text-alert text-[9px] leading-none">&#9650;</span>}
        {!drift && s.is_bottleneck && (
          <span className="text-state-bottleneck text-[9px] leading-none">&#9670;</span>
        )}
      </div>

      <button
        onClick={() => onSelect?.(s.id)}
        title={`${s.id} \u00b7 ${st.label}${s.blind ? " \u00b7 no sensors" : ""} \u00b7 buffer ${s.buffer}/${s.buffer_capacity}`}
        className={[
          "w-[34px] h-[58px] rounded border flex flex-col items-center justify-between",
          "py-1.5 transition-colors",
          border, bg,
          selected ? "ring-1 ring-ink" : "",
          s.blind ? "border-dashed" : "",
        ].join(" ")}
      >
        <span className={`font-mono text-[10px] ${drift || s.is_bottleneck ? codeColour : "text-ink-muted"}`}>
          {s.id.slice(1)}
        </span>
        <span className={`text-[9px] leading-none ${codeColour}`}>
          {s.blind ? "?" : st.glyph}
        </span>
        <span className={`font-mono text-[7px] tracking-wider ${codeColour}`}>{code}</span>
      </button>
    </div>
  );
}

function Legend() {
  const items = [
    ["\u25CF", "RUN", "text-state-run"],
    ["\u25A0", "BLK", "text-state-blocked"],
    ["\u25CB", "STV", "text-state-starved"],
    ["\u2715", "DWN", "text-state-down"],
    ["\u25C6", "BOTTLENECK", "text-state-bottleneck"],
    ["\u25B2", "DRIFT", "text-alert"],
    ["?", "BLIND", "text-ink-faint"],
  ];
  return (
    <div className="flex gap-3 flex-wrap">
      {items.map(([g, l, c]) => (
        <span key={l} className={`font-mono text-[9px] tracking-label ${c}`}>
          {g} {l}
        </span>
      ))}
    </div>
  );
}
