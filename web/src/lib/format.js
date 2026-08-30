export const pct = (v, d = 1) => (v == null ? "\u2014" : `${(100 * v).toFixed(d)}%`);
export const num = (v, d = 1) => (v == null ? "\u2014" : v.toFixed(d));

/**
 * Single source of truth for station state.
 *
 * Each state carries a colour, a three-letter code and a glyph. All three,
 * deliberately: the ribbon has to survive projection, colour blindness, and a
 * photograph of a screen, so colour never carries meaning on its own.
 */
export const STATE_STYLE = {
  active:  { code: "RUN", glyph: "\u25CF", text: "text-state-run",      label: "Running" },
  working: { code: "RUN", glyph: "\u25CF", text: "text-state-run",      label: "Running" },
  blocked: { code: "BLK", glyph: "\u25A0", text: "text-state-blocked",  label: "Blocked" },
  starved: { code: "STV", glyph: "\u25CB", text: "text-state-starved",  label: "Starved" },
  down:    { code: "DWN", glyph: "\u2715", text: "text-state-down",     label: "Down" },
  idle:    { code: "IDL", glyph: "\u00B7", text: "text-ink-faint",      label: "Idle" },
};
export const stateStyle = (s) => STATE_STYLE[s] ?? STATE_STYLE.idle;

/* ------------------------------------------------------------------ */
/* Human-readable names for machine identifiers.
 *
 * Sensor tags and run directories are named for the pipeline, not for a person
 * standing at the line. Rendering the raw identifiers put strings like
 * "s04_electrode_wear_pct" and "nodrift_s11" in front of the user. These map to
 * the plant's own vocabulary; the raw id stays the value passed to the API, so
 * only the display changes.
 */

// Measurement suffix -> label and unit. Keyed on the tag with its station
// prefix removed, so one entry covers all 35 stations.
const MEASURES = {
  clamp_force_N:      ["Clamp force", "N"],
  electrode_wear_pct: ["Electrode wear", "%"],
  gap_left_mm:        ["Left gap", "mm"],
  gap_right_mm:       ["Right gap", "mm"],
  gap_diff_abs:       ["Gap mismatch", null],
  weld_current_A:     ["Weld current", "A"],
  weld_time_ms:       ["Weld time", "ms"],
  booth_temp_C:       ["Booth temperature", "°C"],
  film_thickness_um:  ["Film thickness", "µm"],
  flow_rate_ml_min:   ["Flow rate", "ml/min"],
  humidity_pct:       ["Humidity", "%"],
  angle_deg:          ["Angle", "°"],
  leak_rate_ccm:      ["Leak rate", "ccm"],
  torque_Nm:          ["Torque", "N·m"],
};

/**
 * "s04_electrode_wear_pct" -> "S04 electrode wear (%)"
 * "dwell_S07"              -> "S07 dwell time"
 * Anything unrecognised falls back to spaced words rather than showing raw
 * snake_case, so a new tag degrades gracefully instead of leaking an identifier.
 */
export function featureName(raw) {
  if (!raw) return "";

  // Cycle-time features are named the other way round: dwell_S07.
  const dwell = /^dwell_(S\d+)$/i.exec(raw);
  if (dwell) return `${dwell[1].toUpperCase()} dwell time`;
  if (raw === "dwell_std") return "Dwell spread across stations";

  const station = /^s(\d+)_(.+)$/i.exec(raw);
  if (station) {
    const id = `S${station[1]}`;
    const known = MEASURES[station[2]];
    if (known) {
      const [name, unit] = known;
      return unit ? `${id} ${name.toLowerCase()} (${unit})` : `${id} ${name.toLowerCase()}`;
    }
    return `${id} ${station[2].replace(/_/g, " ")}`;
  }
  return raw.replace(/_/g, " ");
}

// Run directory prefix -> what that family of runs is for. Set in run_all.py:
// "run" seeds carry injected drift and are the evaluation set, "nodrift" seeds
// are the drift-free calibration set, "tune" seeds are for threshold tuning.
const RUN_FAMILIES = {
  run: "Evaluation",
  nodrift: "Calibration",
  tune: "Tuning",
};

/** "nodrift_s11" -> "Calibration 11". Falls back to the raw id, de-underscored. */
export function runName(raw) {
  if (!raw) return "";
  const m = /^([a-z]+)_s(\d+)$/i.exec(raw);
  if (m) {
    const family = RUN_FAMILIES[m[1].toLowerCase()];
    if (family) return `${family} ${Number(m[2])}`;
  }
  return raw.replace(/_/g, " ");
}
