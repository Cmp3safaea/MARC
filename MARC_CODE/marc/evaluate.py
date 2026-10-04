"""Retrospective evaluation against a proxy reference set (paper, Section 3).

Runs the pipeline on an input CSV, joins a reference CSV (one row per
project: student_id, reference = 1 if the project is in the reference set)
and prints every quantity reported in Section 3:

  * the denominators (paired, parseable, R6-scored, R8-scored, grid counts)
  * gap statistics and the gap-bin / band-crossing table behind Figure 2
  * Table 2: selected and conservative operating points, OR of all rules,
    flag-everything, expected 10% random sample, with 95% Wilson intervals
  * the top-k review-budget result under the implemented ordering
  * the two R2 components, the R3/R4 overlap and per-rule flag counts
  * Table 3: joint ablation of R3 and R4
  * Spearman correlation between comment polarity and the midpoint mark

Usage:
    python -m marc.evaluate INPUT.csv REFERENCE.csv [--json metrics.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from .pipeline import Config, PipelineOutput, RULES, run_pipeline, score_flags
from .text_rules import LEXICON_NAME


# ------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(flagged: Sequence[bool], positive: Sequence[bool]) -> dict:
    """TP/FP/FN/TN, precision, recall and F1 with Wilson intervals."""
    tp = sum(1 for f, p in zip(flagged, positive) if f and p)
    fp = sum(1 for f, p in zip(flagged, positive) if f and not p)
    fn = sum(1 for f, p in zip(flagged, positive) if not f and p)
    tn = sum(1 for f, p in zip(flagged, positive) if not f and not p)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and prec + rec > 0 else float("nan"))
    return {"flagged": tp + fp, "n": tp + fp + fn + tn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "F1": f1,
            "precision_ci": wilson(tp, tp + fp),
            "recall_ci": wilson(tp, tp + fn)}


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1     # average rank for ties
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman's rho (Pearson correlation of average ranks)."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if not sxx or not syy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


# ------------------------------------------------------------ evaluation
def read_reference(path: str) -> Dict[str, bool]:
    ref = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("student_id") or "").strip()
            if sid:
                ref[sid] = (row.get("reference") or "").strip().lower() in (
                    "1", "true", "yes", "y")
    return ref


GAP_BINS = (("0", lambda g: g == 0),
            ("1-4", lambda g: 0 < g < 5),
            ("5-9", lambda g: 5 <= g < 10),
            (">=10", lambda g: g >= 10))


def evaluate(out: PipelineOutput, reference: Dict[str, bool]) -> dict:
    """All Section 3 quantities for one cohort, as a plain dictionary."""
    cfg = out.config
    rows = sorted(out.ranked, key=lambda r: r.input_order)
    n = len(rows)
    pos = [reference.get(r.student_id, False) for r in rows]
    n_pos = sum(pos)
    base = n_pos / n if n else float("nan")

    # ---- denominators
    parse = [r for r in rows if r.gap is not None]
    grids = [int(r.fm_grid_present) + int(r.as_grid_present) for r in rows]
    res = {
        "lexicon": LEXICON_NAME,
        "denominators": {
            "paired_projects": n,
            "pairs_with_two_parseable_marks": len(parse),
            "pairs_with_unparseable_mark": n - len(parse),
            "r6_scored": sum(r.r6_scored for r in rows),
            "r8_scored": sum(r.r8_scored for r in rows),
            "expected_grids": 2 * n,
            "populated_grids": sum(grids),
            "projects_both_grids": grids.count(2),
            "projects_one_grid": grids.count(1),
            "projects_no_grid": grids.count(0),
        },
        "reference": {"positives": n_pos, "base_rate": base},
    }

    # ---- gap statistics and Figure 2(b)
    gaps = [r.gap for r in parse]
    crossings = [r for r in parse if r.boundary_crossed]
    res["gaps"] = {
        "mean": statistics.fmean(gaps) if gaps else None,
        "median": statistics.median(gaps) if gaps else None,
        "sd_population": statistics.pstdev(gaps) if gaps else None,
        "sd_sample": statistics.stdev(gaps) if len(gaps) > 1 else None,
        "band_crossings": len(crossings),
        "crossing_rate": len(crossings) / len(parse) if parse else None,
        "crossings_below_tau_gap": sum(1 for r in crossings
                                       if r.gap < cfg.tau_gap),
        "bins": [{"bin": name,
                  "n": sum(1 for r in parse if test(r.gap)),
                  "crossings": sum(1 for r in crossings if test(r.gap))}
                 for name, test in GAP_BINS],
    }

    # ---- Table 2
    any_rule = [any(r.flags.values()) for r in rows]
    k = round(0.10 * n)
    res["table2"] = {
        f"pipeline S>={cfg.cutoff_selected:g}": confusion(
            [r.score >= cfg.cutoff_selected for r in rows], pos),
        f"pipeline S>={cfg.cutoff_conservative:g}": confusion(
            [r.score >= cfg.cutoff_conservative for r in rows], pos),
        "OR of all rules": confusion(any_rule, pos),
        "flag everything": confusion([True] * n, pos),
        "random sample (expected)": {
            "flagged": k, "n": n, "recall": k / n if n else float("nan"),
            "precision": base,
            "F1": (2 * base * (k / n) / (base + k / n)
                   if n and base + k / n > 0 else float("nan"))},
    }

    # ---- review-budget result under the implemented ordering
    top = out.ranked[:k]
    top_tp = sum(1 for r in top if reference.get(r.student_id, False))
    cut_score = top[-1].score if top else None
    res["budget"] = {
        "k": k, "budget": k / n if n else None,
        "true_positives_in_top_k": top_tp,
        "recall": top_tp / n_pos if n_pos else float("nan"),
        "expected_random_recall": k / n if n else None,
        "ties_at_cut": (sum(1 for r in out.ranked if r.score == cut_score)
                        if top else 0),
        "note": "ties keep input-file order; the result depends on that "
                "ordering and is descriptive",
    }

    # ---- rule-level counts
    per_rule = {}
    for rule in RULES:
        fl = [r.flags[rule] for r in rows]
        c = confusion(fl, pos)
        per_rule[rule.upper()] = {"flags": c["flagged"], "TP": c["TP"],
                                  "FP": c["FP"], "precision": c["precision"]}
    res["rules"] = per_rule
    res["r2_components"] = {
        "genuine_crossings": sum(r.r2_component == "crossing" for r in rows),
        "undefined_band": sum(r.r2_component == "undefined" for r in rows),
    }
    res["r3_and_r4_together"] = sum(
        1 for r in rows if r.flags["r3"] and r.flags["r4"])

    # ---- Table 3: joint ablation of R3 and R4
    ablated = [score_flags(r.flags, cfg.weights, exclude=("r3", "r4"))
               >= cfg.cutoff_selected for r in rows]
    full = res["table2"][f"pipeline S>={cfg.cutoff_selected:g}"]
    abl = confusion(ablated, pos)
    abl["delta_F1"] = abl["F1"] - full["F1"]
    res["table3_without_r3_r4"] = abl

    # ---- polarity against the mark
    scored = [r for r in rows if r.r6_scored]
    res["polarity_vs_mark"] = {
        "spearman_rho": spearman([r.midpoint for r in scored],
                                 [r.polarity for r in scored]),
        "n": len(scored)}

    # ---- reference-positive cases below the operating point
    missed = [r for r, p in zip(rows, pos)
              if p and r.score < cfg.cutoff_selected]
    res["missed"] = {
        "below_operating_point": len(missed),
        "no_rule_fired": sum(1 for r in missed if not any(r.flags.values())),
    }
    return res


# --------------------------------------------------------------- report
def _f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _ci(ci):
    return f"[{_f(ci[0])}-{_f(ci[1])}]"


def print_report(res: dict) -> None:
    d = res["denominators"]
    print(f"Lexicon for R6: {res['lexicon']}")
    print("\nDENOMINATORS")
    print(f"  paired projects                 {d['paired_projects']}")
    print(f"  two parseable marks             "
          f"{d['pairs_with_two_parseable_marks']}")
    print(f"  with an unparseable mark        "
          f"{d['pairs_with_unparseable_mark']}")
    print(f"  scored by R6                    {d['r6_scored']}")
    print(f"  scored by R8                    {d['r8_scored']}")
    print(f"  grids populated / expected      "
          f"{d['populated_grids']} / {d['expected_grids']}")
    print(f"  projects with 2 / 1 / 0 grids   {d['projects_both_grids']} / "
          f"{d['projects_one_grid']} / {d['projects_no_grid']}")
    r = res["reference"]
    print(f"  reference-positive              {r['positives']} "
          f"(base rate {_f(r['base_rate'])})")

    g = res["gaps"]
    print("\nGAPS (pairs with two parseable marks)")
    print(f"  mean {_f(g['mean'], 2)}  median {_f(g['median'], 2)}  "
          f"SD {_f(g['sd_population'], 2)} (divisor n) / "
          f"{_f(g['sd_sample'], 2)} (divisor n-1)")
    print(f"  band crossings {g['band_crossings']} "
          f"({_f(g['crossing_rate'])}), of which "
          f"{g['crossings_below_tau_gap']} have a gap below the R1 threshold")
    print("  gap bin   crossings/n")
    for b in g["bins"]:
        print(f"    {b['bin']:<6}  {b['crossings']}/{b['n']}")

    print("\nTABLE 2 - performance against the reference set")
    print(f"  {'Strategy':<27}{'Flagged':>12}  {'Recall':<22}"
          f"{'Precision':<22}F1")
    for name, c in res["table2"].items():
        flagged = f"{c['flagged']} of {c['n']}"
        rec = _f(c["recall"]) + (" " + _ci(c["recall_ci"])
                                 if "recall_ci" in c else "")
        pre = _f(c["precision"]) + (" " + _ci(c["precision_ci"])
                                    if "precision_ci" in c else "")
        print(f"  {name:<27}{flagged:>12}  {rec:<22}{pre:<22}{_f(c['F1'])}")
        if "TP" in c:
            print(f"  {'':<27}{'':>12}  TP {c['TP']}  FP {c['FP']}  "
                  f"FN {c['FN']}  TN {c['TN']}")

    b = res["budget"]
    print(f"\nREVIEW BUDGET - top {b['k']} ({_f(b['budget'])} of the cohort)")
    print(f"  recall {_f(b['recall'])} against {_f(b['expected_random_recall'])}"
          f" expected under random selection")
    print(f"  {b['ties_at_cut']} projects share the score at the cut; "
          f"{b['note']}")

    print("\nRULES")
    print(f"  {'Rule':<6}{'Flags':>7}{'TP':>6}{'FP':>6}  Precision")
    for rule, c in res["rules"].items():
        print(f"  {rule:<6}{c['flags']:>7}{c['TP']:>6}{c['FP']:>6}  "
              f"{_f(c['precision'])}")
    c2 = res["r2_components"]
    print(f"  R2 components: {c2['genuine_crossings']} genuine crossings + "
          f"{c2['undefined_band']} undefined band")
    print(f"  R3 and R4 fire together on {res['r3_and_r4_together']} projects")

    a = res["table3_without_r3_r4"]
    print("\nTABLE 3 - without R3 and R4 together")
    print(f"  recall {_f(a['recall'])}  precision {_f(a['precision'])}  "
          f"F1 {_f(a['F1'])}  delta F1 {a['delta_F1']:+.3f}")
    print(f"  TP {a['TP']}  FP {a['FP']}  FN {a['FN']}  TN {a['TN']}")

    p = res["polarity_vs_mark"]
    print(f"\nPOLARITY v MARK  Spearman rho {_f(p['spearman_rho'])} "
          f"(n = {p['n']})")
    m = res["missed"]
    print(f"\nMISSED  {m['below_operating_point']} reference-positive cases "
          f"fall below the operating point; {m['no_rule_fired']} of them "
          f"trigger no rule")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="marc.evaluate",
        description="Evaluate MARC against a proxy reference set.")
    ap.add_argument("input", help="pipeline input CSV")
    ap.add_argument("reference", help="CSV with columns student_id, reference")
    ap.add_argument("--json", default=None, help="also write metrics as JSON")
    args = ap.parse_args(argv)

    out = run_pipeline(args.input, Config())
    res = evaluate(out, read_reference(args.reference))
    print_report(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nMetrics written to {args.json}")
    print("\nReminder: the reference set records earlier moderation "
          "decisions. These figures measure agreement with those decisions, "
          "not whether a grade is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
