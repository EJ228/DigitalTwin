# Frontend

React + Vite + Tailwind + Recharts. Four views over one twin.

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` and `/stream` to `http://localhost:8000`, so run
the backend first (`make api` from the project root) and open only :5173.
`make dev` from the root starts both.

## Structure

```
src/
  App.jsx                    shell: sidebar, header, replay controls, blind toggle
  lib/
    api.js                   every backend call. Views never fetch directly.
    useTwinStream.js         websocket replay. Controls are sent over the OPEN
                             socket, never by reconnecting, so toggling sensors
                             mid-demo does not drop a frame.
    format.js                STATE_STYLE -- single source of truth for how a
                             station state is coloured, coded and glyphed
  components/
    Sidebar.jsx              nav + live andon list
    StationRibbon.jsx        35 stations, zone-grouped, markers above the cells
    AlertBanner.jsx          the andon banner
    DriftChart.jsx           observed -> threshold -> projection with 90% band
    MetricTile.jsx           label / figure / denominator / progress rule
    CounterfactualPanel.jsx  Monte Carlo rollout table
  views/
    SupervisorView.jsx       "what do I do in the next 30 seconds"
    EscapeWindowView.jsx     the split screen: one fault, two worlds
    ManagerView.jsx          constraint migration band, disruption log
    LeadershipView.jsx       business case, coverage, instrumentation ranking
```

## Design notes

Colour never carries meaning alone. Every station state has a colour, a
three-letter code and a glyph, so the ribbon survives projection, colour
blindness, and a photograph of a screen.

Palette and type live in `tailwind.config.js` under semantic names
(`state.run`, `state.bottleneck`, `alert`, `accent`). Components reference the
names, never hex, so restyling is a single-file change.

Numbers use tabular figures so live metrics do not jitter as digits change.

When a station has no sensors, `DriftChart` does not render an empty chart --
it says what it cannot see. A twin that hides its own ignorance is worse than
no twin.
