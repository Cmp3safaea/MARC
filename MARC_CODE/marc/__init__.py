"""MARC - Moderation Assistance by Rule-based Checking.

Reference implementation of the five-stage screening pipeline described in:

    A. Safaei and I. K. Ihianle, "MARC: A Transparent Framework for
    Automated Grade-Consistency Checking in Assessment Moderation",
    Nottingham Trent University.

MARC identifies records for human review. It does not make grading
decisions and it uses no AI language model.
"""
from .canonicalise import (canonicalise, honours_band, Canonical,
                           MAPPING_VERSION)
from .rules import Project, compute_discrepancy, apply_structural_rules
from .text_rules import apply_text_rules, polarity, fk_grade, LEXICON_NAME
from .pipeline import (Config, Result, MarkerProfile, PipelineOutput,
                       run_projects, run_pipeline, read_projects_csv,
                       write_ranked_csv, write_marker_profiles_csv)

__version__ = "1.1.0"
__all__ = [
    "canonicalise", "honours_band", "Canonical", "MAPPING_VERSION",
    "Project", "compute_discrepancy", "apply_structural_rules",
    "apply_text_rules", "polarity", "fk_grade", "LEXICON_NAME",
    "Config", "Result", "MarkerProfile", "PipelineOutput",
    "run_projects", "run_pipeline", "read_projects_csv",
    "write_ranked_csv", "write_marker_profiles_csv",
]
