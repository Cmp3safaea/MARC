"""Stage 5 and orchestration - priority score, ranking, marker summaries.

S(i) = sum_j w_j * R_j(i). Scores are grouped into Standard (S < 2),
Moderate (2 <= S < 5), High (5 <= S < 8) and Critical (S >= 8). Marker
summaries are developmental only (paper, Sections 2 and 5).
"""
from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .canonicalise import MAPPING_VERSION
from .rules import (Project, Discrepancy, StructuralFlags,
                    compute_discrepancy, apply_structural_rules)
from .text_rules import TextFlags, apply_text_rules

RULES = ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8")


@dataclass
class Config:
    """All thresholds and weights in one place (Table 1 defaults).

    These values were calibrated on the paper's 2025-26 cohort. Re-derive
    them from prior data of your own before operational use.
    """
    tau_gap: float = 5.0          # R1 threshold, percentage points
    tau_chars: int = 200          # R3 threshold, characters
    r6_mad_k: float = 2.0
    r7_grade: float = 6.0
    r8_sd_k: float = 1.5
    weights: Dict[str, float] = field(default_factory=lambda: {
        "r1": 2.0, "r2": 3.0, "r3": 1.0, "r4": 1.5,
        "r5": 3.0, "r6": 2.0, "r7": 1.0, "r8": 1.5,
    })
    cutoff_selected: float = 2.0       # in-sample operating point
    cutoff_conservative: float = 3.0
    # R2(i) = max(b_i, u_i). Set False to count genuine crossings only.
    r2_include_undefined: bool = True
    # R7 only where R3 did not fire. False = combined comment >= tau_chars.
    r7_requires_r3_pass: bool = True
    bands = ((8.0, "Critical"), (5.0, "High"), (2.0, "Moderate"),
             (0.0, "Standard"))

    def band_of(self, score: float) -> str:
        for floor, name in self.bands:
            if score >= floor:
                return name
        return "Standard"


@dataclass
class Result:
    """One ranked project."""
    student_id: str
    input_order: int              # row position in the input (tie-break)
    fm_value: Optional[float]
    as_value: Optional[float]
    fm_confidence: str
    as_confidence: str
    fm_source: str
    as_source: str
    gap: Optional[float]
    signed_gap: Optional[float]
    midpoint: Optional[float]
    boundary_crossed: bool
    band_undefined: bool
    r2_component: str
    flags: Dict[str, bool]
    r6_scored: bool
    r8_scored: bool
    polarity: Optional[float]
    fm_grid_present: bool
    as_grid_present: bool
    score: float
    band: str
    reasons: List[str]


@dataclass
class MarkerProfile:
    """Developmental marker summary - never evaluative."""
    marker: str
    n_marks: int
    median: Optional[float]
    mad: Optional[float]
    leniency_z: Optional[float]
    anomaly_rate: Optional[float]


@dataclass
class PipelineOutput:
    ranked: List[Result]
    marker_profiles: List[MarkerProfile]
    config: Config

    def working_set(self, cutoff: Optional[float] = None) -> List[Result]:
        cut = self.config.cutoff_selected if cutoff is None else cutoff
        return [r for r in self.ranked if r.score >= cut]


def score_flags(flags: Dict[str, bool], weights: Dict[str, float],
                exclude: tuple = ()) -> float:
    """S = sum of the weights of the triggered rules (optionally ablated)."""
    return sum(weights[k] for k in RULES if flags[k] and k not in exclude)


def run_projects(projects: List[Project],
                 cfg: Optional[Config] = None) -> PipelineOutput:
    """Run Stages 1-5 over a list of projects and return ranked output."""
    cfg = cfg or Config()

    # Stages 1 + 2
    discs: List[Discrepancy] = [compute_discrepancy(p) for p in projects]

    # Stage 3
    structural: List[StructuralFlags] = [
        apply_structural_rules(p, d, cfg.tau_gap, cfg.tau_chars,
                               cfg.r2_include_undefined)
        for p, d in zip(projects, discs)]

    # Stage 4 (cohort-relative). One document per project: the two markers'
    # comments concatenated.
    combined = [(p.fm_comment or "") + "\n" + (p.as_comment or "")
                for p in projects]
    midpoints = [d.midpoint for d in discs]
    r7_ok = ([not sf.r3 for sf in structural]
             if cfg.r7_requires_r3_pass else None)
    textual: List[TextFlags] = apply_text_rules(
        combined, midpoints, cfg.tau_chars,
        cfg.r6_mad_k, cfg.r7_grade, cfg.r8_sd_k, r7_ok)

    # Stage 5 - score and rank
    results: List[Result] = []
    for i, (p, d, sf, tf) in enumerate(zip(projects, discs,
                                           structural, textual)):
        flags = {"r1": sf.r1, "r2": sf.r2, "r3": sf.r3, "r4": sf.r4,
                 "r5": sf.r5, "r6": tf.r6, "r7": tf.r7, "r8": tf.r8}
        s = score_flags(flags, cfg.weights)
        results.append(Result(
            student_id=p.student_id, input_order=i,
            fm_value=d.fm.value, as_value=d.asr.value,
            fm_confidence=d.fm.confidence, as_confidence=d.asr.confidence,
            fm_source=d.fm_source, as_source=d.as_source,
            gap=d.gap, signed_gap=d.signed_gap, midpoint=d.midpoint,
            boundary_crossed=d.boundary_crossed,
            band_undefined=d.band_undefined, r2_component=sf.r2_component,
            flags=flags, r6_scored=tf.r6_scored, r8_scored=tf.r8_scored,
            polarity=tf.polarity,
            fm_grid_present=p.fm_grid_present,
            as_grid_present=p.as_grid_present,
            score=s, band=cfg.band_of(s),
            reasons=sf.reasons + tf.reasons))
    # Fixed secondary ordering: equal scores keep their input-file order.
    results.sort(key=lambda r: (-r.score, r.input_order))

    profiles = _marker_profiles(projects, discs, results, cfg)
    return PipelineOutput(results, profiles, cfg)


def _marker_profiles(projects, discs, results, cfg) -> List[MarkerProfile]:
    """Per member of staff: first-marker and assessor marks pooled by name.

    Staff identity is the exact marker string. The leniency z-score is the
    marker's mean minus the mean of all markers' means, divided by the
    population standard deviation (divisor n) of the markers' means. Each
    marker's mean counts equally. No minimum allocation is enforced.
    """
    marks: Dict[str, List[float]] = {}
    flagged: Dict[str, List[bool]] = {}
    by_order = {r.input_order: r for r in results}
    for i, (p, d) in enumerate(zip(projects, discs)):
        for marker, value in ((p.marker_fm, d.fm.value),
                              (p.marker_as, d.asr.value)):
            marker = (marker or "").strip()
            if not marker or value is None:
                continue
            marks.setdefault(marker, []).append(value)
            flagged.setdefault(marker, []).append(
                by_order[i].score >= cfg.cutoff_selected)
    means = [statistics.fmean(v) for v in marks.values() if v]
    cohort_mean = statistics.fmean(means) if means else None
    cohort_sd = statistics.pstdev(means) if len(means) > 1 else None
    out = []
    def natural(item):                    # staff2 before staff10
        return [int(t) if t.isdigit() else t
                for t in re.split(r"(\d+)", item[0])]
    for marker, values in sorted(marks.items(), key=natural):
        med = statistics.median(values)
        mad = statistics.median(abs(v - med) for v in values)
        z = None
        if cohort_mean is not None and cohort_sd:
            z = (statistics.fmean(values) - cohort_mean) / cohort_sd
        fr = flagged.get(marker, [])
        out.append(MarkerProfile(
            marker=marker, n_marks=len(values), median=med, mad=mad,
            leniency_z=z, anomaly_rate=(sum(fr) / len(fr)) if fr else None))
    return out


# ------------------------------------------------------------------ I/O
_FALSE = ("0", "false", "no", "n", "")


def read_projects_csv(path: str) -> List[Project]:
    projects = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            def get(key, default=""):
                value = row.get(key)
                return default if value is None else value
            projects.append(Project(
                student_id=get("student_id").strip(),
                fm_token=get("fm_token"),
                as_token=get("as_token"),
                fm_comment=get("fm_comment"),
                as_comment=get("as_comment"),
                csv_fm_token=get("csv_fm_token"),
                csv_as_token=get("csv_as_token"),
                fm_grid_present=get("fm_grid_present", "1").strip().lower()
                    not in _FALSE,
                as_grid_present=get("as_grid_present", "1").strip().lower()
                    not in _FALSE,
                marker_fm=get("marker_fm"),
                marker_as=get("marker_as"),
            ))
    return projects


def run_pipeline(csv_path: str, cfg: Optional[Config] = None) -> PipelineOutput:
    """Convenience wrapper: CSV in, ranked output object back."""
    return run_projects(read_projects_csv(csv_path), cfg)


def _num(value, fmt="{:.1f}"):
    return "" if value is None else fmt.format(value)


def write_ranked_csv(output: PipelineOutput, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "student_id", "fm_mark", "as_mark",
                    "fm_confidence", "as_confidence",
                    "fm_source", "as_source", "gap", "signed_gap",
                    "midpoint", "boundary_crossed", "band_undefined",
                    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
                    "score", "band", "justification", "mapping_version"])
        for i, r in enumerate(output.ranked, 1):
            w.writerow([
                i, r.student_id, _num(r.fm_value), _num(r.as_value),
                r.fm_confidence, r.as_confidence, r.fm_source, r.as_source,
                _num(r.gap), _num(r.signed_gap, "{:+.1f}"), _num(r.midpoint),
                int(r.boundary_crossed), int(r.band_undefined),
                *[int(r.flags[k]) for k in RULES],
                f"{r.score:.1f}", r.band, " | ".join(r.reasons),
                MAPPING_VERSION])


def write_marker_profiles_csv(output: PipelineOutput, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["marker", "n_marks", "median", "MAD",
                    "leniency_z", "anomaly_rate", "note"])
        for m in output.marker_profiles:
            w.writerow([
                m.marker, m.n_marks, _num(m.median), _num(m.mad),
                _num(m.leniency_z, "{:+.2f}"), _num(m.anomaly_rate, "{:.2f}"),
                "developmental use only; small allocations are unstable"])
