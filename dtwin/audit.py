"""
Anti-circularity audit.

Our substrate is generated, so the first question any reviewer should ask is
whether the models were tuned on the labels. This module answers it mechanically
instead of rhetorically.

The rule: modules that produce predictions may not read truth tables. Only
`dtwin/scoring.py` and the evaluation scripts may.
"""

from __future__ import annotations

import pathlib
import re

# Modules that are allowed to touch ground truth.
TRUTH_READERS = {"scoring.py", "audit.py", "adapters.py", "simulator.py",
                 "schema.py", "engines.py", "twin.py"}

# Modules that must never touch it.
PREDICTORS = {"detectors.py", "reconstruct.py", "spc.py",
              "hazard.py", "blind.py", "coherence.py", "conformal.py",
              "forecast.py", "graphsage.py"}

TRUTH_PATTERN = re.compile(
    r"truth_bottleneck|truth_defects|truth_drift|truth_episodes|\.truth\(|is_defective|cause_station"
)


# --------------------------------------------------------------------------
# Frontend audit
# --------------------------------------------------------------------------
#
# The backend has been under audit from the start; the frontend was not, and it
# quietly accumulated three fabricated figures -- a hardcoded uncertainty band,
# a drawn "forecast" interval, and a literal "Confidence 92%". None was hidden
# in the code, but all three were hidden in the PRODUCT: rendered in the same
# weight and colour as real model output, with nothing to tell a viewer which
# numbers the pipeline produced and which someone typed.
#
# This check closes that gap. A display component may not contain a numeric
# literal presented as a measurement.

FRONTEND_DIR = "web/src"

# Patterns that indicate a fabricated measurement in a display path.
FABRICATION_PATTERNS = [
    (r"[Cc]onfidence\s+\d+(\.\d+)?\s*%", "hardcoded confidence percentage"),
    (r"confidence_band['\"]?\s*[:=]\s*\d", "hardcoded uncertainty band"),
    (r"\bband\s*:\s*(past\s*\?)?\s*\[?\s*p\.v\s*\*", "band computed in the frontend"),
    (r"posterior_sd\s*[:=]\s*\d", "hardcoded posterior width"),
    (r"\baccuracy\s*[:=]\s*0?\.\d", "hardcoded accuracy"),
]

# Declared design constants. These are ALLOWED because they are business
# assumptions, not measurements, and each is labelled as illustrative on screen.
FRONTEND_ALLOWED = {
    "LeadershipView.jsx": ["REWORK_COST", "INCIDENTS_PER_YEAR"],
    "ManagerView.jsx": ["COMPARISON"],      # measured figures, sourced in a comment
    "EscapeWindowView.jsx": ["TAKT_SECONDS"],
}


def assert_no_fabricated_figures(root: str | pathlib.Path | None = None) -> list[str]:
    """Fail if a display component presents a typed-in number as a measurement."""
    base = pathlib.Path(root) if root else pathlib.Path(__file__).parents[1] / FRONTEND_DIR
    if not base.exists():
        return []
    violations = []
    for path in sorted(base.rglob("*.jsx")):
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("//", 1)[0]
            if code.strip().startswith("*"):
                continue                       # doc comment
            for pat, why in FABRICATION_PATTERNS:
                if re.search(pat, code):
                    violations.append(f"{path.name}:{i}: {why}: {line.strip()[:90]}")
    if violations:
        raise AssertionError(
            "Fabricated figure in a display path. Every number shown to a user "
            "must come from the backend.\n  " + "\n  ".join(violations))
    return violations


def assert_no_truth_leak(package_dir: str | pathlib.Path | None = None) -> list[str]:
    """Scan predictor modules for any reference to ground truth.

    Returns a list of violations; empty means clean. Raises on violation so it
    can be wired into the invariant suite and fail the build.
    """
    root = pathlib.Path(package_dir or pathlib.Path(__file__).parent)
    violations = []
    for path in sorted(root.glob("*.py")):
        if path.name not in PREDICTORS:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]          # ignore comments and prose
            if TRUTH_PATTERN.search(code):
                violations.append(f"{path.name}:{i}: {line.strip()}")
    if violations:
        raise AssertionError(
            "Truth leak: a prediction module references ground truth.\n  "
            + "\n  ".join(violations)
        )
    return violations


if __name__ == "__main__":
    assert_no_truth_leak()
    print(f"clean: no truth references in {sorted(PREDICTORS)}")
    assert_no_fabricated_figures()
    print("clean: no fabricated figures in the frontend display path")
