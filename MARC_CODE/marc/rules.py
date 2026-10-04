"""Stages 2 and 3 - discrepancy computation and structural rules R1-R5.

Stage 2 computes, for each paired project i with marks x (first marker) and
y (assessor): the absolute gap g = |x - y|, the signed gap d = y - x, the
midpoint m = (x + y) / 2, the boundary-crossing indicator b and the
missing-band indicator u. Stage 3 applies rules R1-R5 (paper, Section 2,
Table 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .canonicalise import Canonical, canonicalise, honours_band


@dataclass
class Project:
    """One paired project (one input row).

    The first marker and the assessor mark independently on their marking
    grids (fm_token / as_token). They then agree a grade, which is stored in
    the institutional CSV (csv_fm_token / csv_as_token), and the moderator
    confirms the final grade. Rule R5 compares each grid with the CSV record.
    """
    student_id: str
    fm_token: str = ""        # first marker's mark as written on the grid
    as_token: str = ""        # assessor's mark as written on the grid
    fm_comment: str = ""
    as_comment: str = ""
    csv_fm_token: str = ""    # first-marker entry in the institutional CSV
    csv_as_token: str = ""    # assessor entry in the institutional CSV
    fm_grid_present: bool = True
    as_grid_present: bool = True
    marker_fm: str = ""
    marker_as: str = ""


@dataclass
class Discrepancy:
    """Stage 2 output for one project."""
    fm: Canonical
    asr: Canonical
    gap: Optional[float]          # g = |x - y|; None if either mark is unparseable
    signed_gap: Optional[float]   # d = y - x (assessor minus first marker)
    midpoint: Optional[float]     # m = (x + y) / 2
    boundary_crossed: bool        # b: both parseable and different bands
    band_undefined: bool          # u: either mark unparseable
    fm_source: str = "grid"       # where the Stage 2 mark came from
    as_source: str = "grid"


def _stage2_mark(grid_token: str, csv_token: str) -> Tuple[Canonical, str]:
    """Mark used by Stage 2 for one marker role.

    A mark written on the grid is used directly. Where the grid carries no
    mark at all (grid missing or left as a blank template because the mark
    was recorded centrally), the institutional CSV entry is used. A grid
    token that is present but unparseable is NOT replaced: it stays
    unparseable so that the record is surfaced.
    """
    if (grid_token or "").strip():
        return canonicalise(grid_token), "grid"
    if (csv_token or "").strip():
        return canonicalise(csv_token), "csv"
    return canonicalise(grid_token), "none"


def compute_discrepancy(p: Project) -> Discrepancy:
    fm, fm_src = _stage2_mark(p.fm_token, p.csv_fm_token)
    asr, as_src = _stage2_mark(p.as_token, p.csv_as_token)
    if fm.value is None or asr.value is None:
        # Pairs containing an unparseable mark stay available for the
        # structural checks but are excluded from the gap statistics.
        return Discrepancy(fm, asr, None, None, None, False, True,
                           fm_src, as_src)
    gap = abs(fm.value - asr.value)
    signed = asr.value - fm.value
    mid = (fm.value + asr.value) / 2.0
    crossed = honours_band(fm.value) != honours_band(asr.value)
    return Discrepancy(fm, asr, gap, signed, mid, crossed, False,
                       fm_src, as_src)


@dataclass
class StructuralFlags:
    """Stage 3 output: rules R1-R5 with human-readable reasons."""
    r1: bool = False
    r2: bool = False
    r3: bool = False
    r4: bool = False
    r5: bool = False
    r2_component: str = ""        # "crossing", "undefined" or ""
    reasons: list = field(default_factory=list)


def apply_structural_rules(p: Project, d: Discrepancy,
                           tau_gap: float = 5.0,
                           tau_chars: int = 200,
                           r2_include_undefined: bool = True
                           ) -> StructuralFlags:
    """Apply R1-R5 to one project. Every triggered rule appends a reason."""
    f = StructuralFlags()

    # R1 - large gap: g >= tau_gap percentage points
    if d.gap is not None and d.gap >= tau_gap:
        f.r1 = True
        f.reasons.append(f"R1: gap {d.gap:.1f} pp >= {tau_gap:g} pp")

    # R2 - band discrepancy: R2(i) = max(b_i, u_i).
    # The two components are recorded separately.
    if d.boundary_crossed:
        f.r2 = True
        f.r2_component = "crossing"
        f.reasons.append(
            f"R2: markers give different honours bands "
            f"({honours_band(d.fm.value)} vs {honours_band(d.asr.value)})")
    elif d.band_undefined and r2_include_undefined:
        f.r2 = True
        f.r2_component = "undefined"
        bad = [f"{side} {c.token!r} ({c.note})"
               for side, c in (("first-marker", d.fm), ("assessor", d.asr))
               if c.value is None]
        f.reasons.append("R2: honours band undefined - " + "; ".join(bad))

    # R3 - sparse rationale: either comment shorter than tau_chars characters
    fm_len = len((p.fm_comment or "").strip())
    as_len = len((p.as_comment or "").strip())
    if min(fm_len, as_len) < tau_chars:
        f.r3 = True
        f.reasons.append(
            f"R3: rationale too short (first marker {fm_len}, assessor "
            f"{as_len} characters; threshold {tau_chars})")

    # R4 - missing artefact: grid absent on disk or left unfilled
    if not p.fm_grid_present or not p.as_grid_present:
        f.r4 = True
        missing = [side for side, ok in
                   (("first-marker", p.fm_grid_present),
                    ("assessor", p.as_grid_present)) if not ok]
        f.reasons.append("R4: grid missing or template-default ("
                         + ", ".join(missing) + ")")

    # R5 - grid band differs from the institutional CSV band (either marker)
    for side, grid_tok, csv_tok in (
            ("first-marker", p.fm_token, p.csv_fm_token),
            ("assessor", p.as_token, p.csv_as_token)):
        if not (grid_tok or "").strip() or not (csv_tok or "").strip():
            continue
        g = canonicalise(grid_tok)
        c = canonicalise(csv_tok)
        if g.value is None or c.value is None:
            continue
        if honours_band(g.value) != honours_band(c.value):
            f.r5 = True
            f.reasons.append(
                f"R5: {side} grid band {honours_band(g.value)} differs from "
                f"CSV band {honours_band(c.value)}")
    return f
