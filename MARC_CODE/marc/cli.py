"""Command-line interface: run Stages 1-5 on an input CSV.

Usage:
    python -m marc INPUT.csv [--out ranked.csv] [--markers markers.csv]
        [--tau-gap 5] [--tau-chars 200] [--cutoff 2] [--r2-undefined on|off]
"""
from __future__ import annotations

import argparse
import logging
import sys

from .canonicalise import MAPPING_VERSION
from .pipeline import (Config, run_pipeline, write_ranked_csv,
                       write_marker_profiles_csv)
from .text_rules import LEXICON_NAME


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="marc",
        description="MARC: rank paired projects for moderation review "
                    "(assistance, not automation).")
    ap.add_argument("input", help="input CSV (see README for the columns)")
    ap.add_argument("--out", default="ranked.csv",
                    help="ranked output CSV (default: ranked.csv)")
    ap.add_argument("--markers", default=None,
                    help="optional marker-summary CSV (developmental only)")
    ap.add_argument("--tau-gap", type=float, default=5.0,
                    help="R1 gap threshold in percentage points (default 5)")
    ap.add_argument("--tau-chars", type=int, default=200,
                    help="R3 comment-length threshold in characters "
                         "(default 200)")
    ap.add_argument("--cutoff", type=float, default=2.0,
                    help="operating point on the priority score (default 2)")
    ap.add_argument("--r2-undefined", choices=("on", "off"), default="on",
                    help="include the undefined-band component of R2 "
                         "(default on)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")

    cfg = Config(tau_gap=args.tau_gap, tau_chars=args.tau_chars,
                 cutoff_selected=args.cutoff,
                 r2_include_undefined=args.r2_undefined == "on")
    out = run_pipeline(args.input, cfg)

    write_ranked_csv(out, args.out)
    if args.markers:
        write_marker_profiles_csv(out, args.markers)

    n = len(out.ranked)
    print(f"Grade mapping version {MAPPING_VERSION}; R6 lexicon: "
          f"{LEXICON_NAME}")
    print(f"{n} paired projects processed.")
    print(f"{len(out.working_set())} in the working set "
          f"(score >= {cfg.cutoff_selected:g}).")
    for band in ("Critical", "High", "Moderate", "Standard"):
        k = sum(1 for r in out.ranked if r.band == band)
        print(f"  {band:9s} {k}")
    print(f"Ranked list written to {args.out}")
    if args.markers:
        print(f"Marker summaries written to {args.markers} "
              f"(developmental use only)")
    print("Reminder: this output is a sorting aid. "
          "The final decision on every case stays with the moderator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
