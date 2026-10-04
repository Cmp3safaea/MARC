"""Unit tests. Several are tied directly to figures stated in the paper."""
import csv
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)

from marc import (canonicalise, honours_band, Project, Config,  # noqa: E402
                  run_projects, fk_grade, polarity)
from marc.rules import compute_discrepancy, apply_structural_rules  # noqa: E402
from marc.text_rules import apply_text_rules  # noqa: E402
from marc.evaluate import wilson, confusion, evaluate, spearman  # noqa: E402
from marc.heldout import analyse, grade_changed  # noqa: E402

LONG = ("The project presents a thorough investigation of the problem. "
        "The methodology is rigorous and the design decisions are justified. "
        "The evaluation is convincing and the limitations are discussed. "
        "The report is clearly written and coherent throughout.")


class TestCanonicalise(unittest.TestCase):
    """Section 2 and Appendix A."""

    def test_numeric(self):
        self.assertEqual(canonicalise("65").value, 65.0)
        self.assertEqual(canonicalise("65%").confidence, "Exact")
        self.assertEqual(canonicalise("0").value, 0.0)   # genuine zero mark

    def test_table_a1(self):
        expected = {"3LOW": 41.7, "3MID": 45.0, "3HIGH": 48.3,
                    "22LOW": 51.7, "22MID": 55.0, "22HIGH": 58.3,
                    "21LOW": 61.7, "21MID": 65.0, "21HIGH": 68.3,
                    "1LOW": 71.7, "1MID": 75.0, "1HIGH": 78.3}
        for token, value in expected.items():
            c = canonicalise(token)
            self.assertAlmostEqual(c.value, value)
            self.assertEqual(c.confidence, "Banded")

    def test_labels(self):
        for token, value in {"First": 75.0, "1st": 75.0, "2:1": 65.0,
                             "2:2": 55.0, "Third": 45.0, "3rd": 45.0,
                             "Fail": 20.0}.items():
            c = canonicalise(token)
            self.assertEqual(c.value, value)
            self.assertEqual(c.confidence, "Label")

    def test_admin_codes_are_unparseable(self):
        for token in ("", "-", "NS", "NEC", "SAI", "N/A", "ZERO", None):
            c = canonicalise(token)
            self.assertIsNone(c.value)
            self.assertEqual(c.confidence, "Missing")

    def test_out_of_range_is_unparseable_not_clamped(self):
        for token in ("105", "650", "-5", "100.5%"):
            c = canonicalise(token)
            self.assertIsNone(c.value, token)
            self.assertEqual(c.note, "number outside [0, 100]")
        self.assertEqual(canonicalise("100").value, 100.0)

    def test_1exc_has_no_value(self):
        self.assertIsNone(canonicalise("1EXC").value)

    def test_bands(self):
        self.assertEqual(honours_band(39.9), "Fail")
        self.assertEqual(honours_band(40), "Third")
        self.assertEqual(honours_band(59.9), "2:2")
        self.assertEqual(honours_band(68), "2:1")
        self.assertEqual(honours_band(70), "First")
        self.assertIsNone(honours_band(None))


class TestStage2(unittest.TestCase):
    def test_gap_signed_gap_midpoint(self):
        d = compute_discrepancy(Project("student1", "68", "72"))
        self.assertEqual(d.gap, 4.0)
        self.assertEqual(d.signed_gap, 4.0)       # d = y - x
        self.assertEqual(d.midpoint, 70.0)
        self.assertTrue(d.boundary_crossed)
        self.assertFalse(d.band_undefined)

    def test_mark_falls_back_to_csv_when_grid_has_none(self):
        d = compute_discrepancy(Project("student1", "", "64",
                                        csv_fm_token="21MID"))
        self.assertEqual(d.fm.value, 65.0)
        self.assertEqual(d.fm_source, "csv")
        self.assertEqual(d.gap, 1.0)

    def test_unparseable_grid_token_is_not_replaced(self):
        d = compute_discrepancy(Project("student1", "1EXC", "72",
                                        csv_fm_token="1HIGH"))
        self.assertIsNone(d.gap)
        self.assertTrue(d.band_undefined)


class TestStructuralRules(unittest.TestCase):
    def flags(self, p, **kw):
        return apply_structural_rules(p, compute_discrepancy(p), **kw)

    def test_r1_threshold(self):
        self.assertTrue(self.flags(Project("s", "60", "65")).r1)
        self.assertFalse(self.flags(Project("s", "60", "64")).r1)

    def test_r2_is_max_of_crossing_and_undefined(self):
        f = self.flags(Project("s", "68", "72"))
        self.assertEqual((f.r2, f.r2_component), (True, "crossing"))
        f = self.flags(Project("s", "NS", "72"))
        self.assertEqual((f.r2, f.r2_component), (True, "undefined"))
        f = self.flags(Project("s", "NS", "72"), r2_include_undefined=False)
        self.assertFalse(f.r2)
        self.assertFalse(self.flags(Project("s", "62", "64")).r2)

    def test_undefined_pair_scores_5_5_with_r3_and_r4(self):
        """Section 3: the undefined-band pairs score 5.5 (R2 + R3 + R4),
        and 2.5 without R2."""
        p = Project("s", "NS", "NS", fm_grid_present=False,
                    as_grid_present=False)
        self.assertEqual(run_projects([p]).ranked[0].score, 5.5)
        cfg = Config(r2_include_undefined=False)
        self.assertEqual(run_projects([p], cfg).ranked[0].score, 2.5)

    def test_r3_either_comment_short(self):
        self.assertTrue(self.flags(Project("s", "60", "61", LONG, "ok")).r3)
        self.assertFalse(self.flags(Project("s", "60", "61", LONG, LONG)).r3)

    def test_r5_band_mismatch(self):
        f = self.flags(Project("s", "68", "72", csv_fm_token="1LOW",
                               csv_as_token="1LOW"))
        self.assertTrue(f.r5)
        f = self.flags(Project("s", "68", "72", csv_fm_token="21HIGH",
                               csv_as_token="1LOW"))
        self.assertFalse(f.r5)


class TestWorkedExample(unittest.TestCase):
    """Table 4: 68 against 72, template-default first-marker grid,
    62-character assessor rationale, CSV 1LOW. R2, R3, R4, R5; S = 8.5."""

    def test_table_4(self):
        p = Project(
            "student1", fm_token="68", as_token="72", fm_comment="",
            as_comment="Good project with a working prototype. "
                       "Testing was acceptable.",
            csv_fm_token="1LOW", csv_as_token="1LOW",
            fm_grid_present=False, as_grid_present=True,
            marker_fm="staff1", marker_as="staff2")
        self.assertEqual(len(p.as_comment), 62)
        r = run_projects([p]).ranked[0]
        self.assertEqual(r.gap, 4.0)
        fired = [k.upper() for k, v in r.flags.items() if v]
        self.assertEqual(fired, ["R2", "R3", "R4", "R5"])
        self.assertAlmostEqual(r.score, 8.5)
        self.assertEqual(r.band, "Critical")

    def test_banded_tokens_give_same_rules(self):
        p = Project("student1", "21HIGH", "1LOW", "", "short",
                    csv_fm_token="1LOW", csv_as_token="1LOW",
                    fm_grid_present=False)
        r = run_projects([p]).ranked[0]
        self.assertAlmostEqual(r.gap, 3.4)
        self.assertAlmostEqual(r.score, 8.5)


class TestScoreBands(unittest.TestCase):
    def test_bands(self):
        cfg = Config()
        self.assertEqual(cfg.band_of(1.5), "Standard")
        self.assertEqual(cfg.band_of(2.0), "Moderate")
        self.assertEqual(cfg.band_of(4.5), "Moderate")
        self.assertEqual(cfg.band_of(5.0), "High")
        self.assertEqual(cfg.band_of(8.0), "Critical")

    def test_table_1_weights(self):
        self.assertEqual(Config().weights, {
            "r1": 2.0, "r2": 3.0, "r3": 1.0, "r4": 1.5,
            "r5": 3.0, "r6": 2.0, "r7": 1.0, "r8": 1.5})


class TestTextRules(unittest.TestCase):
    def test_polarity_divides_by_all_tokens(self):
        self.assertEqual(polarity(""), 0.0)
        p = polarity("good the the the")          # one scored word in four
        self.assertGreater(p, 0)
        self.assertAlmostEqual(p * 4, polarity("good"))

    def test_fk_formula(self):
        # 4 words, 1 sentence, 4 syllables
        self.assertAlmostEqual(fk_grade("The cat sat down."),
                               0.39 * 4 + 11.8 * 1 - 15.59)
        self.assertIsNone(fk_grade("123 ... !!!"))

    def test_r7_only_when_r3_passes(self):
        simple = "It is good. " * 30
        long_ok = run_projects([Project("s", "60", "61", simple, simple)])
        self.assertTrue(long_ok.ranked[0].flags["r7"])
        one_short = run_projects([Project("s", "60", "61", simple, "ok")])
        self.assertTrue(one_short.ranked[0].flags["r3"])
        self.assertFalse(one_short.ranked[0].flags["r7"])

    def test_r6_and_r8_need_a_cohort(self):
        flags = apply_text_rules(["good work"] * 3, [60.0, 61.0, 62.0])
        self.assertFalse(any(f.r6 or f.r8 for f in flags))
        self.assertFalse(any(f.r6_scored for f in flags))


class TestRanking(unittest.TestCase):
    def test_ties_keep_input_order(self):
        ps = [Project(f"student{i}", "60", "61", LONG, LONG)
              for i in range(1, 6)]
        ps[3] = Project("student4", "60", "72", LONG, LONG)
        out = run_projects(ps)
        self.assertEqual([r.student_id for r in out.ranked],
                         ["student4", "student1", "student2", "student3",
                          "student5"])

    def test_marker_summaries_pool_roles(self):
        ps = [Project("student1", "60", "70", marker_fm="staff1",
                      marker_as="staff2"),
              Project("student2", "50", "62", marker_fm="staff2",
                      marker_as="staff1")]
        prof = {m.marker: m for m in run_projects(ps).marker_profiles}
        self.assertEqual(prof["staff1"].n_marks, 2)
        self.assertEqual(prof["staff1"].median, 61.0)
        self.assertEqual(prof["staff1"].anomaly_rate, 1.0)


class TestPaperArithmetic(unittest.TestCase):
    """The reported figures follow from the reported confusion counts."""

    def test_table_2_selected_point(self):
        tp, fp, fn = 249, 6, 17
        p, r = tp / (tp + fp), tp / (tp + fn)
        self.assertAlmostEqual(p, 0.976, 3)
        self.assertAlmostEqual(r, 0.936, 3)
        self.assertAlmostEqual(2 * p * r / (p + r), 0.956, 3)
        lo, hi = wilson(249, 255)
        self.assertEqual((round(lo, 3), round(hi, 3)), (0.950, 0.989))
        lo, hi = wilson(249, 266)
        self.assertEqual((round(lo, 3), round(hi, 3)), (0.900, 0.960))

    def test_table_3_ablation(self):
        tp, fp, fn = 222, 6, 44
        p, r = tp / (tp + fp), tp / (tp + fn)
        self.assertAlmostEqual(r, 0.835, 3)
        self.assertAlmostEqual(p, 0.974, 3)
        self.assertAlmostEqual(2 * p * r / (p + r), 0.899, 3)

    def test_table_5_intervals(self):
        self.assertEqual(tuple(round(v, 2) for v in wilson(5, 96)),
                         (0.02, 0.12))
        self.assertEqual(tuple(round(v, 2) for v in wilson(5, 16)),
                         (0.14, 0.56))
        self.assertEqual(tuple(round(v, 2) for v in wilson(14, 115)),
                         (0.07, 0.19))
        self.assertEqual(tuple(round(v, 2) for v in wilson(14, 26)),
                         (0.35, 0.71))

    def test_confusion(self):
        c = confusion([True, True, False, False], [True, False, True, False])
        self.assertEqual((c["TP"], c["FP"], c["FN"], c["TN"]), (1, 1, 1, 1))

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


class TestSampleData(unittest.TestCase):
    """The shipped sample data runs end to end and is fully anonymous."""
    root = Path(__file__).resolve().parents[1] / "sample_data"

    def test_identifiers_are_synthetic(self):
        with open(self.root / "sample_cohort.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["student_id"] for r in rows],
                         [f"student{i}" for i in range(1, len(rows) + 1)])
        for r in rows:
            self.assertRegex(r["marker_fm"], r"^staff\d+$")
            self.assertRegex(r["marker_as"], r"^staff\d+$")

    def test_evaluate_runs(self):
        from marc import run_pipeline
        from marc.evaluate import read_reference
        out = run_pipeline(str(self.root / "sample_cohort.csv"))
        res = evaluate(out, read_reference(
            str(self.root / "sample_reference.csv")))
        self.assertEqual(res["denominators"]["paired_projects"], 60)
        self.assertEqual(out.ranked[0].band, "Critical")
        worked = next(r for r in out.ranked if r.student_id == "student1")
        self.assertAlmostEqual(worked.score, 8.5)
        full = res["table2"]["pipeline S>=2"]
        self.assertEqual(full["TP"] + full["FP"] + full["FN"] + full["TN"], 60)

    def test_heldout_runs(self):
        from marc.heldout import load_grade_record
        rows = load_grade_record(str(self.root / "sample_grade_record.csv"))
        on = analyse(rows)["summary"]
        off = analyse(rows, r2_include_undefined=False)["summary"]
        self.assertEqual(on["n"], 80)
        self.assertEqual(on["selected"]["flagged"] - off["selected"]["flagged"],
                         on["unparseable_pairs"])

    def test_grade_changed(self):
        base = {"agreed": "21MID", "moderator": "21MID", "final": "21MID"}
        self.assertFalse(grade_changed(base))
        self.assertTrue(grade_changed({**base, "moderator": "21HIGH"}))
        self.assertTrue(grade_changed({**base, "final": "1LOW"}))
        self.assertFalse(grade_changed({**base, "moderator": ""}))


if __name__ == "__main__":
    unittest.main()
