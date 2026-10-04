"""Tests for the grid scanner (skipped unless python-docx is installed)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import docx
    HAVE_DOCX = True
except ImportError:
    HAVE_DOCX = False


@unittest.skipUnless(HAVE_DOCX, "python-docx not installed")
class TestGridScanner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_grid(self, folder, name, lines):
        d = docx.Document()
        for line in lines:
            d.add_paragraph(line)
        target = self.root / folder
        target.mkdir(exist_ok=True)
        d.save(str(target / name))

    def test_filled_grid_is_scanned(self):
        from marc.grid_scanner import scan_folder
        self._make_grid("staff1", "student1_first_marker.docx", [
            "Project marking grid",
            "Overall grade: 21HIGH",
            "The project shows a solid design and a working prototype. "
            "The evaluation chapter is honest about limitations.",
        ])
        r = scan_folder(str(self.root))[0]
        self.assertEqual(r.student_id, "STUDENT1")
        self.assertEqual(r.role, "FM")
        self.assertEqual(r.token, "21HIGH")
        self.assertTrue(r.filled)
        self.assertIn("solid design", r.comment)

    def test_blank_template_is_detected(self):
        from marc.grid_scanner import scan_folder
        self._make_grid("staff2", "student2_assessor.docx", [
            "Project marking grid", "Enter your comments here"])
        r = scan_folder(str(self.root))[0]
        self.assertFalse(r.filled)
        self.assertEqual(r.role, "AS")
        self.assertEqual(r.token, "")

    def test_grade_without_rationale_is_template_default(self):
        """Table 4: a grid can carry a grade and still be template-default."""
        from marc.grid_scanner import scan_folder
        self._make_grid("staff1", "student3_first.docx", [
            "Project marking grid", "Overall grade: 21HIGH"])
        r = scan_folder(str(self.root))[0]
        self.assertEqual(r.token, "21HIGH")
        self.assertFalse(r.filled)

    def test_role_uses_whole_words(self):
        from marc.grid_scanner import _detect_role
        self.assertEqual(_detect_role(Path("student4_thomas_first.docx")), "FM")
        self.assertEqual(_detect_role(Path("student4_AS.docx")), "AS")

    def test_build_input_csv(self):
        from marc.grid_scanner import build_input_csv
        self._make_grid("staff1", "student5_first.docx", [
            "Overall grade: 68%",
            "A clear and well organised project with honest evaluation."])
        self._make_grid("staff2", "student5_assessor.docx", [
            "Overall grade: 1LOW",
            "Strong technical work, the report reads well throughout."])
        out = self.root / "input.csv"
        self.assertEqual(build_input_csv(str(self.root), None, str(out)), 1)
        text = out.read_text()
        for expected in ("STUDENT5", "68", "1LOW", "staff1", "staff2"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
