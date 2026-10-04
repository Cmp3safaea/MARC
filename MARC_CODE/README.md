# MARC — Moderation Assistance by Rule-based Checking

Reference implementation of the five-stage screening pipeline described in:

> A. Safaei and I. K. Ihianle, *MARC: A Transparent Framework for Automated
> Grade-Consistency Checking in Assessment Moderation*, Nottingham Trent
> University.

In a double-marked module the first marker and the assessor mark each project
independently, agree a mark, and a moderator checks the outcome. MARC runs the
same consistency check on **every** project and ranks the cases that most
deserve the moderator's attention. It does not grade students, it does not
change grades, and it uses no AI language model. Every flag traces to one
written rule and, where relevant, one threshold.

## Quick start

Python 3.10 or later. From this folder:

```bash
pip install afinn            # AFINN-111 word list, needed for rule R6

# 1. Rank the sample cohort
python -m marc sample_data/sample_cohort.csv --out ranked.csv --markers markers.csv

# 2. Evaluate against the sample reference set (Tables 2 and 3 of the paper)
python -m marc.evaluate sample_data/sample_cohort.csv sample_data/sample_reference.csv

# 3. Structural core against actual grade changes (Table 5 of the paper)
python -m marc.heldout sample_data/sample_grade_record.csv

# 4. Tests and runtime check
python -m unittest discover tests
python benchmark.py
```

If `afinn` is not installed the pipeline still runs, but R6 uses a small
built-in word list and prints a warning. Results obtained that way are not
comparable with the paper.

## Sample data

Everything in `sample_data/` is synthetic, produced by a seeded generator
(`sample_data/make_sample_data.py`). It contains no real student or staff
data: students are `student1`…`student60` and staff are `staff1`…`staff12`.

| File | Used by | Content |
|---|---|---|
| `sample_cohort.csv` | `marc`, `marc.evaluate` | 60 paired projects |
| `sample_reference.csv` | `marc.evaluate` | which projects are in the proxy reference set |
| `sample_grade_record.csv` | `marc.heldout` | 80-row institutional grade record |

`student1` is the paper's worked example (Table 4): 68 against 72, R2, R3, R4
and R5 fire, S = 8.5, Critical. The cohort also includes administrative codes
(`NS`, `ZERO`), an unmapped token (`1EXC`, `FMARG`) and an out-of-range number
(`105`) so that every Stage 1 path is exercised.

The sample figures illustrate the output format only. They are not the
paper's results.

## The five stages

| Stage | What happens |
|---|---|
| 1 | Convert each mark token (`65%`, `2:1`, `22HIGH`, `ZERO`…) to a percentage or to "unparseable", with a confidence label (Exact, Banded, Label, Missing) |
| 2 | Compute the gap, signed gap (assessor minus first marker), midpoint, band-crossing indicator and missing-band indicator |
| 3 | Apply the structural rules R1–R5 |
| 4 | Apply the text checks R6–R8 |
| 5 | Sum the weights of the triggered rules, rank the cohort, build marker summaries |

## The eight rules (Table 1 defaults)

| ID | Rule | What it checks | Weight |
|---|---|---|---|
| R1 | Large gap | gap ≥ 5 pp | 2.0 |
| R2 | Band discrepancy | different honours bands, or either band undefined | 3.0 |
| R3 | Sparse rationale | either comment < 200 characters | 1.0 |
| R4 | Missing artefact | grid absent on disk or unfilled | 1.5 |
| R5 | Grid–CSV band mismatch | grid band differs from CSV band | 3.0 |
| R6 | Sentiment–grade mismatch | centred residual > 2 MAD of cohort | 2.0 |
| R7 | Low-complexity rationale | readability < grade 6 despite length | 1.0 |
| R8 | Comment lexical drift | TF-IDF similarity 1.5 SD below cohort | 1.5 |

Score groups: Standard (S < 2), Moderate (2 ≤ S < 5), High (5 ≤ S < 8),
Critical (S ≥ 8). Selected operating point S ≥ 2; conservative alternative
S ≥ 3. All thresholds and weights live in `Config` (`marc/pipeline.py`) and
were calibrated on one cohort. Re-derive them from your own prior data before
operational use.

## Input

One CSV row per paired project:

| Column | Meaning |
|---|---|
| `student_id` | project identifier |
| `fm_token`, `as_token` | first-marker / assessor mark as written on the grid |
| `fm_comment`, `as_comment` | each marker's written rationale |
| `csv_fm_token`, `csv_as_token` | the entries in the institutional CSV |
| `fm_grid_present`, `as_grid_present` | `1` if the grid exists and is filled, else `0` |
| `marker_fm`, `marker_as` | staff identifiers, for marker summaries |

A mark written on the grid is used directly. Where a grid carries no mark,
Stage 2 takes the mark from the institutional CSV entry and records the
source (`fm_source`, `as_source`) in the output.

To build this file from a folder of `.docx` marking grids (optional, needs
`pip install python-docx`):

```bash
python -m marc.grid_scanner GRIDS_ROOT --institutional-csv marking_report.csv \
    --out pipeline_input.csv --report
```

Grid templates differ between institutions. Adapt the patterns at the top of
`marc/grid_scanner.py` and spot-check scanned grids by hand.

## Output

`ranked.csv` gives, per project: canonical marks and confidence labels, mark
source, gap, signed gap, midpoint, the eight rule flags, priority score, score
group, a plain-text justification for every flag, and the mapping version.
Projects with equal scores keep their input-file order, so the ranking is
reproducible for a given input file.

`markers.csv` gives, per member of staff: number of marks, median, MAD,
leniency z-score and anomaly rate. First-marker and assessor marks are pooled
under the same identifier. These summaries are for developmental
conversations only. They must not be used to change grades or for probation,
appraisal or performance management, and small allocations give unstable
estimates.

## Where each paper statement lives in the code

| Paper | Code |
|---|---|
| Stage 1, Appendix A, Table A1 | `marc/canonicalise.py` |
| Stage 2; R1–R5 | `marc/rules.py` |
| R6, R7, R8 | `marc/text_rules.py` |
| Stage 5, score groups, marker summaries | `marc/pipeline.py` |
| Section 3 denominators, gap statistics, Figure 2(b) bins, Tables 2 and 3, review-budget result, per-rule counts, Spearman ρ | `marc/evaluate.py` |
| Section 4, Table 5 | `marc/heldout.py` |
| Runtime | `benchmark.py` |
| Table 4 worked example; Wilson intervals of Tables 2, 3 and 5 | `tests/test_marc.py` |

## Settings the paper leaves to the code

The paper notes several settings that the archived methods record does not
state. This implementation fixes them as follows.

| Setting | Value here |
|---|---|
| Tokenisation (R6, R7, R8) | lower-case runs of letters and apostrophes, `[a-z']+`; hyphenated and multiword lexicon entries are never matched |
| Sentiment lexicon | AFINN-111 (2,477 entries), read explicitly from the `afinn` package's data folder. The package's default English list is the later AFINN-en-165 |
| R8 `n` and document frequency | computed over every project in the cohort |
| R8 standard deviation | divisor n |
| Leniency z-score standard deviation | divisor n |
| Staff identity across marker roles | exact match on the identifier string |
| Secondary ordering of equal scores | input-file order |
| Mapping version | `MAPPING_VERSION` in `marc/canonicalise.py`, written to every output row |

## Changes from the earlier `mod-proj` code

1. R2 now fires when either band is undefined, as defined in Section 2
   (R2 = max(b, u)). The two components are reported separately.
   `--r2-undefined off` restores genuine crossings only.
2. A number outside [0, 100] is returned as unparseable and logged. It was
   previously clamped.
3. The signed gap is assessor minus first marker.
4. R6 reads AFINN-111 explicitly, and warns if it has to fall back.
5. R7 is applied only where R3 did not fire.
6. A grid that carries a grade but no written rationale counts as
   template-default (Table 4).
7. Stage 2 takes the mark from the institutional CSV where the grid has none.
8. The anomaly rate follows the configured operating point.
9. Evaluation code for Tables 2, 3 and 5 is included.
10. The runtime check times one run including file input.

## Known limits

- `1EXC` and the fail-band tokens `FMARG` and `FMID` have no percentage and no
  band in the current mapping, so a pair containing one is excluded from gap
  statistics and triggers the undefined-band component of R2.
- **Table 5 as published was produced with the undefined-band component off.**
  Run `python -m marc.heldout … --r2-undefined off` to reproduce it. With the
  default (on), more projects are flagged.
- R8's eligible set here is every project with a parseable midpoint and a
  non-empty comment, in a band of at least five documents. The paper reports
  300 eligible projects; the rule that produced that count is not recorded.
- The paper's case-study data are not included, so the figures in Tables 2
  and 3 cannot be regenerated from this folder alone.
- The R8 centroid includes the scored document. A leave-one-out sensitivity
  analysis is still outstanding.

## Licence

MIT (see `LICENSE`).
