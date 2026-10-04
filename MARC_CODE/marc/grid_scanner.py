"""Stage 0 (optional) - scan .docx marking grids and build the pipeline input.

Walks a folder tree of .docx marking grids (one sub-folder per marker),
extracts from each grid the grade token and the free-text comments, detects
grids left as blank templates, and merges everything with the institutional
CSV (the agreed-grade record) into the input CSV that the pipeline reads.

Where a grid is missing or blank, its mark columns are left empty and the
pipeline takes the Stage 2 mark from the institutional CSV entry instead.

Requires the optional dependency python-docx (pip install python-docx).

IMPORTANT: marking-grid templates differ between institutions. The patterns
below match the case-study template and common UK conventions. Adapt
GRADE_PATTERNS, ROLE_HINTS_*, TEMPLATE_PHRASES and ID_PATTERN to your own
template, and spot-check a sample of scanned grids by hand. Linking CSV
marker entries to grid folders is a record-linkage step: linkage errors
show up as R4 or R5 flags.

Usage:
    python -m marc.grid_scanner GRIDS_ROOT \
        --institutional-csv marking_report.csv --out pipeline_input.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------- patterns
# Student identifier inside a file name (adapt to your institution).
ID_PATTERN = re.compile(r"(student\d+|[A-Z]?\d{7,9})", re.IGNORECASE)

# Grade tokens, searched on grade-keyword lines first, then anywhere.
GRADE_PATTERNS = [
    re.compile(r"\b(1EXC|1LOW|1MID|1HIGH|1ST|21LOW|21MID|21HIGH|22LOW|22MID|"
               r"22HIGH|3LOW|3MID|3HIGH|3RD|FMARG|FMID|ZERO|NS|SAI|NEC|FAIL)\b",
               re.I),
    re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%"),
    re.compile(r"\b(2:1|2:2)\b"),
]
GRADE_KEYWORDS = re.compile(r"(overall|agreed|final|grade|mark)\s*[:\-]?",
                            re.IGNORECASE)

# File-name words that identify the marker role (whole words only).
ROLE_HINTS_FM = {"first", "fm", "supervisor", "1st"}
ROLE_HINTS_AS = {"assessor", "second", "2nd", "as"}

# Phrases that appear in the unfilled template (adapt to your template).
TEMPLATE_PHRASES = (
    "enter your comments here", "insert comment", "type here",
    "click here to enter text", "guidance for markers",
)

# A grid with less written rationale than this is template-default, even if
# it carries a grade token.
MIN_FILLED_CHARS = 30


@dataclass
class GridRecord:
    """What the scanner extracted from one .docx marking grid."""
    path: str
    marker: str                  # marker folder name
    student_id: Optional[str]
    role: str                    # "FM", "AS" or "?"
    token: str                   # grade token found, "" if none
    comment: str                 # free text written by the marker
    filled: bool                 # False for blank / template-default grids
    notes: List[str] = field(default_factory=list)


def _extract_text(path: Path) -> List[str]:
    """All paragraph and table-cell texts of a .docx, in document order."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "grid scanning needs python-docx: pip install python-docx"
        ) from exc
    d = docx.Document(str(path))
    texts = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    for tbl in d.tables:
        for row in tbl.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t:
                    texts.append(t)
    return texts


def _find_grade(texts: List[str]) -> str:
    """Prefer a token on a line with a grade keyword; fall back to anywhere."""
    keyword_lines = [t for t in texts if GRADE_KEYWORDS.search(t)]
    for pool in (keyword_lines, texts):
        for line in pool:
            for pat in GRADE_PATTERNS:
                m = pat.search(line)
                if m:
                    return m.group(1).upper()
    return ""


def _detect_role(path: Path) -> str:
    words = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    if words & ROLE_HINTS_AS:
        return "AS"
    if words & ROLE_HINTS_FM:
        return "FM"
    return "?"


def _marker_comment(texts: List[str]) -> str:
    """Free text written by the marker: template boilerplate, headings and
    short labels are dropped; sentence-like lines are kept."""
    out = []
    for t in texts:
        low = t.lower()
        if any(p in low for p in TEMPLATE_PHRASES):
            continue
        if GRADE_KEYWORDS.search(t) and len(t) < 40:
            continue                      # grade label line, not commentary
        sentence_like = len(t) >= 40 or (len(t) >= 25 and "." in t)
        if not sentence_like:
            continue                      # headings, labels, template titles
        out.append(t)
    return "\n".join(out)


def scan_grid(path: Path, marker: str) -> GridRecord:
    texts = _extract_text(path)
    token = _find_grade(texts)
    comment = _marker_comment(texts)
    filled = len(comment) >= MIN_FILLED_CHARS
    m = ID_PATTERN.search(path.stem)
    rec = GridRecord(
        path=str(path), marker=marker,
        student_id=m.group(1).upper() if m else None,
        role=_detect_role(path), token=token,
        comment=comment, filled=filled)
    if not token:
        rec.notes.append("no grade token found")
    if not filled:
        rec.notes.append("template-default or blank")
    return rec


def scan_folder(root: str) -> List[GridRecord]:
    """Walk marker sub-folders of `root` and scan every .docx grid."""
    root_path = Path(root)
    records = []
    for path in sorted(root_path.rglob("*.docx")):
        if path.name.startswith("~$"):     # Word lock files
            continue
        marker = path.parent.name if path.parent != root_path else ""
        records.append(scan_grid(path, marker))
    return records


# ------------------------------------------------------------- CSV building
def build_input_csv(grids_root: str, institutional_csv: Optional[str],
                    out_csv: str) -> int:
    """Scan grids, merge with the institutional CSV, write pipeline input.

    The institutional CSV must contain the columns student_id, fm_grade and
    as_grade; fm_marker and as_marker are used when present (adapt the names
    below to your export). Returns the number of paired-project rows written.
    """
    records = scan_folder(grids_root)
    by_student: Dict[str, Dict[str, GridRecord]] = {}
    for rec in records:
        if not rec.student_id:
            continue
        slot = by_student.setdefault(rec.student_id, {})
        key = rec.role if rec.role in ("FM", "AS") else (
            "FM" if "FM" not in slot else "AS")
        # keep the filled grid when duplicates or templates exist
        if key not in slot or (rec.filled and not slot[key].filled):
            slot[key] = rec

    agreed: Dict[str, Dict[str, str]] = {}
    if institutional_csv:
        with open(institutional_csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                sid = (row.get("student_id") or "").strip().upper()
                if sid:
                    agreed[sid] = {
                        "fm": row.get("fm_grade") or "",
                        "as": row.get("as_grade") or "",
                        "fm_marker": row.get("fm_marker") or "",
                        "as_marker": row.get("as_marker") or ""}

    students = sorted(set(by_student) | set(agreed))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["student_id", "fm_token", "as_token", "fm_comment",
                    "as_comment", "csv_fm_token", "csv_as_token",
                    "fm_grid_present", "as_grid_present",
                    "marker_fm", "marker_as"])
        for sid in students:
            slot = by_student.get(sid, {})
            fm, asr = slot.get("FM"), slot.get("AS")
            rec_csv = agreed.get(sid, {})
            w.writerow([
                sid,
                fm.token if fm else "", asr.token if asr else "",
                fm.comment if fm else "", asr.comment if asr else "",
                rec_csv.get("fm", ""), rec_csv.get("as", ""),
                int(bool(fm and fm.filled)), int(bool(asr and asr.filled)),
                rec_csv.get("fm_marker") or (fm.marker if fm else ""),
                rec_csv.get("as_marker") or (asr.marker if asr else "")])
    return len(students)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="marc.grid_scanner",
        description="Scan .docx marking grids and build the pipeline input CSV.")
    ap.add_argument("grids_root", help="folder tree of marking grids "
                                       "(one sub-folder per marker)")
    ap.add_argument("--institutional-csv", default=None,
                    help="institutional CSV with the agreed-grade record")
    ap.add_argument("--out", default="pipeline_input.csv")
    ap.add_argument("--report", action="store_true",
                    help="print a per-grid scan summary")
    args = ap.parse_args(argv)

    if args.report:
        records = scan_folder(args.grids_root)
        filled = sum(1 for r in records if r.filled)
        print(f"{len(records)} grid files scanned; "
              f"{filled} filled, {len(records) - filled} template/blank.")
        for r in records:
            flag = "OK " if r.filled else "..."
            print(f"  [{flag}] {r.role:2s} {r.student_id or '???'} "
                  f"{r.token or '-':7s} {Path(r.path).name}")
    n = build_input_csv(args.grids_root, args.institutional_csv, args.out)
    print(f"{n} paired-project rows written to {args.out}")
    print("Check a sample by hand: grid templates differ between "
          "institutions, and the extraction patterns may need adapting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
