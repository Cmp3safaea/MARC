"""Stage 1 - grade-token canonicalisation (paper, Section 2 and Appendix A).

Maps every mark token a UK marker may write (65, 65%, 2:1, 1st, 22HIGH,
ZERO, NS, ...) to a percentage in [0, 100] or to "unparseable", together
with a confidence label. The recogniser is deterministic and is applied in
the order given in Appendix A:

    1. administrative / non-grade code  -> unparseable (Missing)
    2. number with optional % in [0,100] -> that value   (Exact)
       number outside [0, 100]          -> unparseable (Missing), logged
    3. banded token (Table A1)          -> fixed within-band position (Banded)
    4. classification label             -> band midpoint (Label)
    5. anything else                    -> unparseable (Missing)

Band thresholds and band tokens follow the provider's grade-based marking
descriptors (paper ref. [9]).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("marc.canonicalise")

# Version of the numeric mapping below. Bump it whenever a value changes.
MAPPING_VERSION = "1.1.0"

# Confidence labels
EXACT = "Exact"
BANDED = "Banded"
LABEL = "Label"
MISSING = "Missing"

# Administrative and non-grade codes (Appendix A). ZERO records absence,
# non-submission or work of no assessable merit; a genuine mark of zero is
# entered numerically ("0") and maps to 0.
_ADMIN_CODES = {"", "-", "NS", "NEC", "SAI", "N/A", "ZERO"}

# Banded tokens -> midpoints of the three equal sub-intervals of each
# ten-point class band (Table A1).
_BANDED = {
    "3LOW": 41.7, "3MID": 45.0, "3HIGH": 48.3,
    "22LOW": 51.7, "22MID": 55.0, "22HIGH": 58.3,
    "21LOW": 61.7, "21MID": 65.0, "21HIGH": 68.3,
    "1LOW": 71.7, "1MID": 75.0, "1HIGH": 78.3,
}

# Classification labels -> band midpoints (Appendix A).
_LABELS = {
    "FIRST": 75.0, "1ST": 75.0,
    "2:1": 65.0,
    "2:2": 55.0,
    "THIRD": 45.0, "3RD": 45.0,
    "FAIL": 20.0,
}

_warned: set = set()     # out-of-range tokens already logged

_NUMERIC = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$")


@dataclass(frozen=True)
class Canonical:
    """Result of canonicalising one token."""
    value: Optional[float]   # percentage in [0, 100], or None (unparseable)
    confidence: str          # Exact | Banded | Label | Missing
    token: str               # the original token, stripped
    note: str = ""           # why a token is Missing, for the audit report


def canonicalise(token: Optional[str]) -> Canonical:
    """Token -> (value, confidence). See the module docstring for the order."""
    raw = "" if token is None else str(token).strip()
    upper = raw.upper()

    if upper in _ADMIN_CODES:
        return Canonical(None, MISSING, raw, "administrative code")

    m = _NUMERIC.match(raw)
    if m:
        value = float(m.group(1))
        if 0.0 <= value <= 100.0:
            return Canonical(value, EXACT, raw)
        # Appendix A: never silently adjusted; returned unparseable and logged.
        if raw not in _warned:
            _warned.add(raw)
            log.warning("mark %r is outside [0, 100]; treated as unparseable - "
                        "check it by hand", raw)
        return Canonical(None, MISSING, raw, "number outside [0, 100]")

    if upper in _BANDED:
        return Canonical(_BANDED[upper], BANDED, raw)

    if upper in _LABELS:
        return Canonical(_LABELS[upper], LABEL, raw)

    # Includes 1EXC and the fail-band tokens (FMARG, FMID), which the current
    # mapping gives no percentage and no band. See README, "Known limits".
    return Canonical(None, MISSING, raw, "unrecognised token")


def honours_band(value: Optional[float]) -> Optional[str]:
    """Institutional honours band of a percentage (None if unparseable).

    Fail below 40, Third 40-49, 2:2 50-59, 2:1 60-69, First 70 and above.
    """
    if value is None:
        return None
    if value < 40:
        return "Fail"
    if value < 50:
        return "Third"
    if value < 60:
        return "2:2"
    if value < 70:
        return "2:1"
    return "First"
