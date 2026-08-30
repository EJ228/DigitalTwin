// All backend access lives here. Views never call fetch directly, so pointing
// the app at a different host is a one-line change.

const json = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
};

export const getLine = () => json("/api/line");
export const getRuns = () => json("/api/runs");
export const getTimeline = (run, blind = "") =>
  json(`/api/timeline?run=${run}&blind=${blind}`);
export const getEscapeWindow = (run, blind = "") =>
  json(`/api/escape-window?run=${run}&blind=${blind}`);
export const getRanking = (run, blind = "") =>
  json(`/api/instrumentation-ranking?run=${run}&blind=${blind}`);
export const getEngines = (run) => json(`/api/engines?run=${run}`);
// Cheap readiness probe. /api/engines can take minutes for a run that has no
// completed offline evaluation, so the view checks this first.
export const getEnginesAvailability = (run) =>
  json(`/api/engines/availability?run=${run}`);
export const getCoherence = (run) => json(`/api/coherence-series?run=${run}`);
export const postCounterfactual = (body) =>
  json("/api/counterfactual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export function streamUrl({ run, t, speed, blind }) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const p = new URLSearchParams({ run, t, speed, blind });
  return `${proto}://${location.host}/stream?${p}`;
}
