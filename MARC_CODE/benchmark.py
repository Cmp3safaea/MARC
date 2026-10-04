#!/usr/bin/env python3
"""Runtime check for the complete pipeline (paper, Section 3).

For each cohort size, a synthetic cohort is written to a CSV file and ONE
end-to-end run is timed: reading the file, Stages 1-5 and all eight rules,
including the TF-IDF fit. This is a single observed run, not an averaged
benchmark. The machine specification is printed with the result.

The cohorts are synthetic (student1..N, staff1..55). Absolute times depend
on the machine; they should grow roughly linearly with cohort size.

Usage:
    python benchmark.py            # cohorts of 376, 1000 and 5000
    python benchmark.py 376 2000   # custom cohort sizes
"""
from __future__ import annotations

import csv
import os
import platform
import random
import sys
import tempfile
import time

from marc import run_pipeline, LEXICON_NAME

POSITIVE = ["excellent", "clear", "thorough", "strong", "good", "solid",
            "impressive", "effective", "convincing", "rigorous"]
NEGATIVE = ["weak", "limited", "unclear", "poor", "insufficient",
            "superficial", "missing", "flawed", "shallow", "incomplete"]
NEUTRAL = ["the", "project", "report", "chapter", "design",
           "implementation", "evaluation", "testing", "literature",
           "methodology", "results", "analysis", "student", "work", "system"]


def _comment(rng: random.Random, mark: float) -> str:
    words = []
    for _ in range(rng.randint(40, 120)):
        if rng.random() < 0.08:
            words.append(rng.choice(POSITIVE if mark >= 60 else NEGATIVE))
        else:
            words.append(rng.choice(NEUTRAL))
    sentences = []
    while words:
        k = rng.randint(8, 15)
        sentences.append(" ".join(words[:k]).capitalize() + ".")
        words = words[k:]
    return " ".join(sentences)


def write_cohort(path: str, n: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["student_id", "fm_token", "as_token", "fm_comment",
                    "as_comment", "csv_fm_token", "csv_as_token",
                    "fm_grid_present", "as_grid_present",
                    "marker_fm", "marker_as"])
        for i in range(n):
            base = rng.gauss(60, 10)
            fm = max(30, min(85, round(base + rng.gauss(0, 2))))
            asr = max(30, min(85, round(base + rng.gauss(0, 3))))
            w.writerow([f"student{i + 1}", fm, asr,
                        _comment(rng, fm), _comment(rng, asr), fm, asr,
                        int(rng.random() > 0.3), int(rng.random() > 0.3),
                        f"staff{i % 55 + 1}", f"staff{(i * 7) % 55 + 1}"])


def bench(n: int, folder: str) -> None:
    path = os.path.join(folder, f"cohort_{n}.csv")
    write_cohort(path, n)
    t0 = time.perf_counter()
    out = run_pipeline(path)
    elapsed = time.perf_counter() - t0
    print(f"n={n:5d}: {elapsed * 1000:8.0f} ms, single run including file "
          f"input ({elapsed / n * 1000:.2f} ms/project), "
          f"working set {len(out.working_set())}")


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [376, 1000, 5000]
    print(f"Machine: {platform.platform()}; {platform.machine()}; "
          f"{os.cpu_count()} logical cores")
    print(f"Python {platform.python_version()}; R6 lexicon: {LEXICON_NAME}")
    print("Single core, synthetic cohorts, one run per size.")
    with tempfile.TemporaryDirectory() as folder:
        for n in sizes:
            bench(n, folder)


if __name__ == "__main__":
    main()
