const NAV = [
  { id: "supervisor", label: "Live line" },
  { id: "escape", label: "Escape window" },
  { id: "manager", label: "Shift log" },
  { id: "engines", label: "Model audit" },
  { id: "leadership", label: "Rollout case" },
];

export default function Sidebar({ tab, setTab, andons, supervisor = "M. Okafor \u00b7 Line 2" }) {
  return (
    <aside className="w-[228px] shrink-0 bg-card rounded-l-2xl flex flex-col p-5">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
          <span className="w-3 h-3 rounded-full border-2 border-white" />
        </div>
        <div className="text-[17px] font-semibold tracking-tight">
          Twin<span className="text-ink-faint font-normal">.line</span>
        </div>
      </div>

      <div className="label mt-8 text-accent">Floor</div>
      <nav className="mt-3 space-y-1">
        {NAV.map((n) => (
          <button
            key={n.id}
            onClick={() => setTab(n.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2.5
              ${tab === n.id ? "bg-accent/[0.08] text-ink font-medium" : "text-ink-muted hover:bg-page"}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${tab === n.id ? "bg-accent" : "border border-ink-faint"}`} />
            {n.label}
          </button>
        ))}
      </nav>

      <div className="label mt-8 text-accent">Open andons</div>
      <div className="mt-3 space-y-2">
        {andons.length === 0 && <div className="text-xs text-ink-faint px-1">None</div>}
        {andons.map((a) => (
          <div key={a.key} className="flex items-center gap-2.5 px-1 text-[13px]">
            <span className={`w-2 h-2 shrink-0 ${a.shape} ${a.colour}`} />
            <span className="text-ink-muted">{a.text}</span>
          </div>
        ))}
      </div>

      <div className="mt-auto pt-6">
        <div className="bg-page rounded-lg px-4 py-3">
          <div className="label">Supervisor</div>
          <div className="text-sm mt-1">{supervisor}</div>
        </div>
      </div>
    </aside>
  );
}
