#!/usr/bin/env python3
"""Generate the synthetic sample data shipped with MARC.

Everything here is invented by a seeded random generator. No file contains
real student or staff data: students are student1..N, staff are staff1..N.

Writes, next to this script:
    sample_cohort.csv        pipeline input (one row per paired project)
    sample_reference.csv     proxy reference set for marc.evaluate
    sample_grade_record.csv  institutional grade record for marc.heldout

Usage:  python sample_data/make_sample_data.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_PROJECTS = 60
N_STAFF = 12
N_GRADE_RECORD = 80

STRONG = [
    "The project presents an excellent and thorough investigation of the chosen problem.",
    "The literature review is comprehensive and engages critically with recent research.",
    "The methodology is rigorous and the design decisions are clearly justified.",
    "The implementation is robust, well structured and impressive in scope.",
    "The evaluation is convincing and the limitations are discussed with good insight.",
    "The report is professionally written and the argument is coherent throughout.",
    "Testing is systematic and the results support the conclusions effectively.",
]
MIDDLE = [
    "The project addresses a reasonable problem and the aims are stated clearly.",
    "The literature review covers the main sources although the analysis is descriptive.",
    "The design chapter explains the architecture but several decisions lack justification.",
    "The implementation meets the core requirements and the prototype works as described.",
    "The evaluation is adequate but relies on a small number of test cases.",
    "The report is readable although some sections would benefit from tighter structure.",
    "The conclusion summarises the work and identifies sensible future improvements.",
]
WEAK = [
    "The project shows limited engagement with the literature and the review is superficial.",
    "The design is unclear and important decisions are missing from the documentation.",
    "The implementation is incomplete and several required features fail in the demonstration.",
    "The evaluation is weak and the testing evidence is insufficient to support the claims.",
    "The report is poorly structured and contains errors that make the argument confusing.",
    "The methodology is inadequate and the analysis lacks critical reflection.",
    "The conclusion is shallow and does not address the stated objectives.",
]
SIMPLE = [
    "It is good.", "The work is fine.", "The code runs.", "The test is ok.",
    "I like the plan.", "The aim is met.", "It was a fair job.",
    "The demo ran well.", "The text is clear.", "It is all there.",
]


def to_band_token(mark: int) -> str:
    """Percentage -> banded token (Table A1); marks below 40 stay numeric."""
    if mark < 40:
        return str(mark)
    cls = {4: "3", 5: "22", 6: "21"}.get(mark // 10, "1")
    pos = mark % 10 if mark < 80 else 9
    return cls + ("LOW" if pos < 10 / 3 else "MID" if pos < 20 / 3 else "HIGH")


def to_token(rng: random.Random, mark: int) -> str:
    """Write a mark the way a marker might: 65, 65%, 21MID or 2:1."""
    r = rng.random()
    if r < 0.45:
        return str(mark)
    if r < 0.60:
        return f"{mark}%"
    if r < 0.92 or mark < 40:
        return to_band_token(mark)
    return {4: "3rd", 5: "2:2", 6: "2:1"}.get(mark // 10, "1st")


def comment(rng: random.Random, mark: int, style: str) -> str:
    if style == "empty":
        return ""
    if style == "short":
        return rng.choice(["Good project overall.", "Agreed with first marker.",
                           "Meets the requirements. Sound work.",
                           "See grid for details."])
    if style == "simple":
        return " ".join(rng.choice(SIMPLE) for _ in range(rng.randint(22, 30)))
    pool = STRONG if mark >= 68 else WEAK if mark < 52 else MIDDLE
    if style == "mismatch":          # tone that does not fit the mark
        pool = WEAK if mark >= 60 else STRONG
    return " ".join(rng.sample(pool, rng.randint(4, 6)))


def clamp(x: float) -> int:
    return int(max(30, min(88, round(x))))


def make_cohort(rng: random.Random):
    rows, reference = [], []
    for i in range(1, N_PROJECTS + 1):
        sid = f"student{i}"
        base = rng.gauss(62, 9)
        fm = clamp(base + rng.gauss(0, 2))
        asr = clamp(base + rng.gauss(0, 2))
        if rng.random() < 0.18:                      # a real disagreement
            asr = clamp(fm + rng.choice([-1, 1]) * rng.randint(6, 14))
        marker_fm = f"staff{rng.randint(1, N_STAFF)}"
        marker_as = f"staff{rng.randint(1, N_STAFF)}"
        while marker_as == marker_fm:
            marker_as = f"staff{rng.randint(1, N_STAFF)}"

        fm_grid = rng.random() > 0.22
        as_grid = rng.random() > 0.22
        styles = []
        for present in (fm_grid, as_grid):
            if not present:
                styles.append("empty")
            else:
                r = rng.random()
                styles.append("short" if r < 0.12 else "simple" if r < 0.20
                              else "mismatch" if r < 0.27 else "full")
        fm_token = to_token(rng, fm) if fm_grid else ""
        as_token = to_token(rng, asr) if as_grid else ""

        # institutional CSV: the per-marker record held centrally
        csv_fm, csv_as = to_band_token(fm), to_band_token(asr)
        mismatch = fm_grid and rng.random() < 0.08   # grid and CSV disagree
        if mismatch:
            csv_fm = to_band_token(clamp(fm + rng.choice([-10, 10])))

        row = {
            "student_id": sid, "fm_token": fm_token, "as_token": as_token,
            "fm_comment": comment(rng, fm, styles[0]),
            "as_comment": comment(rng, asr, styles[1]),
            "csv_fm_token": csv_fm, "csv_as_token": csv_as,
            "fm_grid_present": int(fm_grid), "as_grid_present": int(as_grid),
            "marker_fm": marker_fm, "marker_as": marker_as,
        }
        concern = (abs(fm - asr) >= 5 or fm // 10 != asr // 10 or mismatch
                   or not (fm_grid and as_grid) or "short" in styles)
        in_reference = rng.random() < (0.9 if concern else 0.12)
        rows.append(row)
        reference.append({"student_id": sid, "reference": int(in_reference)})

    # The paper's worked example (Table 4): 68 against 72, first-marker grid
    # template-default, 62-character assessor rationale, CSV says 1LOW.
    rows[0].update(
        fm_token="68", as_token="72", fm_comment="",
        as_comment="Good project with a working prototype. Testing was acceptable.",
        csv_fm_token="1LOW", csv_as_token="1LOW",
        fm_grid_present=0, as_grid_present=1)
    reference[0]["reference"] = 1

    # Administrative codes, an unmapped token and an out-of-range number.
    for idx, (fm_tok, as_tok) in {
            9: ("NS", "NS"), 19: ("1EXC", "1HIGH"), 29: ("65", "ZERO"),
            39: ("105", "64"), 49: ("FMARG", "42")}.items():
        rows[idx].update(fm_token=fm_tok, as_token=as_tok,
                         csv_fm_token=fm_tok, csv_as_token=as_tok,
                         fm_grid_present=1, as_grid_present=1)
        reference[idx]["reference"] = 1
    return rows, reference


def make_grade_record(rng: random.Random):
    rows = []
    for i in range(1, N_GRADE_RECORD + 1):
        base = rng.gauss(61, 9)
        sup = clamp(base + rng.gauss(0, 2.5))
        sec = clamp(base + rng.gauss(0, 2.5))
        if rng.random() < 0.15:
            sec = clamp(sup + rng.choice([-1, 1]) * rng.randint(6, 13))
        agreed = clamp((sup + sec) / 2)
        disagree = abs(sup - sec) >= 5 or sup // 10 != sec // 10
        changed = rng.random() < (0.22 if disagree else 0.03)
        moderated = clamp(agreed + rng.choice([-5, 5])) if changed else agreed
        rows.append({
            "Student ID": f"student{i}",
            "Supervisor Grade": to_band_token(sup),
            "Second Marker Grade": to_band_token(sec),
            "Agreed Grade": to_band_token(agreed),
            "Moderator Grade": to_band_token(moderated),
            "Final Grade": to_band_token(moderated),
        })
    for idx, (sup_tok, sec_tok) in {
            4: ("NS", "NS"), 14: ("1EXC", "1HIGH"), 24: ("FMARG", "3LOW"),
            34: ("NEC", "NEC")}.items():
        rows[idx].update({"Supervisor Grade": sup_tok,
                          "Second Marker Grade": sec_tok,
                          "Agreed Grade": sup_tok, "Moderator Grade": sup_tok,
                          "Final Grade": sup_tok})
    return rows


def write(path: Path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    cohort, reference = make_cohort(random.Random(2026))
    write(HERE / "sample_cohort.csv", cohort)
    write(HERE / "sample_reference.csv", reference)
    write(HERE / "sample_grade_record.csv",
          make_grade_record(random.Random(2025)))


if __name__ == "__main__":
    main()
