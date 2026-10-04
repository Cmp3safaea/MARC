"""Stage 4 - the three text checks R6, R7 and R8 (paper, Section 2).

All three are deterministic. No language model and no external service.

R6  sentiment: AFINN polarity of the combined comment (sum of word scores
    divided by the total number of tokens), regressed on the pair midpoint
    across the cohort by OLS with an intercept. A project is flagged when
    |e - median(e)| > 2 * MAD(e), MAD unscaled.
R7  readability: Flesch-Kincaid grade level below 6, applied only to
    comments long enough to pass R3.
R8  similarity: custom TF-IDF vector of the combined comment compared by
    cosine similarity with the centroid of its honours band (self included);
    flagged when more than 1.5 SD below the mean of all scored documents.

Settings the paper leaves to the code, fixed here:
  * tokenisation is the lower-case pattern [a-z']+ for all three rules, so
    hyphenated and multiword lexicon entries are never matched;
  * TF-IDF n and document frequency are computed over every project in the
    cohort (not within band, not only the eligible documents);
  * the R8 standard deviation uses divisor n (population SD).
"""
from __future__ import annotations

import math
import os
import re
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .canonicalise import honours_band

# ---------------------------------------------------------------- AFINN
# Used ONLY if the afinn package is not installed. Results obtained with it
# are not comparable with the paper, and the run says so loudly.
_FALLBACK_LEXICON: Dict[str, int] = {
    "excellent": 3, "outstanding": 5, "superb": 5, "great": 3, "good": 3,
    "strong": 2, "impressive": 3, "clear": 1, "thorough": 2, "solid": 2,
    "well": 2, "comprehensive": 2, "innovative": 2, "effective": 2,
    "robust": 2, "convincing": 2, "rigorous": 2, "insightful": 3,
    "commendable": 3, "professional": 2, "detailed": 1, "coherent": 2,
    "poor": -2, "weak": -2, "limited": -1, "unclear": -2, "confusing": -2,
    "insufficient": -2, "inadequate": -2, "superficial": -2, "missing": -2,
    "wrong": -2, "error": -2, "errors": -2, "fail": -2, "fails": -2,
    "failed": -2, "failure": -2, "lacking": -2, "lacks": -2, "flawed": -2,
    "incomplete": -1, "disappointing": -2, "trivial": -1, "shallow": -2,
}

_WORD = re.compile(r"[a-z']+")


def _load_lexicon() -> Tuple[Dict[str, int], str]:
    """Load AFINN-111 from the afinn package's data folder.

    The package's default English list is AFINN-en-165, a later and larger
    list, so the AFINN-111 file is read explicitly to match the paper.
    """
    try:
        import afinn  # type: ignore
        path = os.path.join(os.path.dirname(afinn.__file__), "data",
                            "AFINN-111.txt")
        lex: Dict[str, int] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "\t" in line:
                    word, score = line.rstrip("\n").rsplit("\t", 1)
                    lex[word] = int(score)
        if lex:
            try:
                from importlib.metadata import version
                package = f"afinn {version('afinn')}"
            except Exception:
                package = "afinn, version not recorded"
            return lex, f"AFINN-111 ({len(lex)} entries, {package})"
    except Exception:
        pass
    warnings.warn(
        "afinn package not found: R6 is using a small built-in fallback "
        "word list, NOT AFINN-111. Install it with 'pip install afinn' to "
        "match the paper.", RuntimeWarning, stacklevel=2)
    return dict(_FALLBACK_LEXICON), (
        f"FALLBACK word list ({len(_FALLBACK_LEXICON)} entries) - "
        f"not AFINN-111")


_LEXICON, LEXICON_NAME = _load_lexicon()


def tokens(text: str) -> List[str]:
    """Lower-case runs of letters and apostrophes."""
    return _WORD.findall((text or "").lower())


def polarity(text: str) -> float:
    """Sum of AFINN word scores divided by the TOTAL number of tokens."""
    words = tokens(text)
    if not words:
        return 0.0
    return sum(_LEXICON.get(w, 0) for w in words) / len(words)


# ------------------------------------------------------- Flesch-Kincaid
_VOWELS = "aeiouy"


def _syllables(word: str) -> int:
    """Vowel-group heuristic with a silent final-e adjustment."""
    word = word.lower().strip("'")
    if not word:
        return 0
    count, prev = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev:
            count += 1
        prev = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def fk_grade(text: str) -> Optional[float]:
    """0.39 * (words per sentence) + 11.8 * (syllables per word) - 15.59.

    Sentences are maximal runs of . ? ! with a minimum count of one.
    A comment with no words receives no score (None).
    """
    words = tokens(text)
    if not words:
        return None
    sentences = max(len(re.findall(r"[.!?]+", text or "")), 1)
    syll = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syll / len(words)) - 15.59


# ----------------------------------------------------------- TF-IDF / R8
def _tfidf_matrix(texts: Sequence[str]) -> List[Dict[str, float]]:
    """(term count / document length) * log((1 + n) / (1 + document freq)).

    Not the scikit-learn default: no +1 after the logarithm and no L2
    normalisation. No stop-word list and no minimum document frequency.
    """
    docs = [tokens(t) for t in texts]
    n = len(docs)
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))
    out: List[Dict[str, float]] = []
    for doc in docs:
        tf = Counter(doc)
        length = max(len(doc), 1)
        out.append({w: (c / length) * math.log((1 + n) / (1 + df[w]))
                    for w, c in tf.items()})
    return out


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    num = sum(a[w] * b[w] for w in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0          # a zero vector scores cosine zero
    return num / (na * nb)


# ------------------------------------------------------------- interface
@dataclass
class TextFlags:
    r6: bool = False
    r7: bool = False
    r8: bool = False
    r6_scored: bool = False   # entered the R6 fit (the R6 denominator)
    r8_scored: bool = False   # received an R8 similarity (the R8 denominator)
    polarity: Optional[float] = None
    reasons: list = field(default_factory=list)


def apply_text_rules(comments: Sequence[str],
                     midpoints: Sequence[Optional[float]],
                     tau_chars: int = 200,
                     r6_mad_k: float = 2.0,
                     r7_grade: float = 6.0,
                     r8_sd_k: float = 1.5,
                     r7_eligible: Optional[Sequence[bool]] = None
                     ) -> List[TextFlags]:
    """Apply R6-R8 across a whole cohort (R6 and R8 are cohort-relative).

    comments[i]    combined first-marker + assessor comment of project i
    midpoints[i]   Stage 2 midpoint (None if unparseable)
    r7_eligible[i] True when project i passed R3 (both comments long
                   enough). If omitted, the combined length is used.
    """
    n = len(comments)
    flags = [TextFlags() for _ in range(n)]

    # ---- R6: OLS of polarity on midpoint; flag |e - median| > k * MAD ----
    scored = [i for i in range(n)
              if midpoints[i] is not None and tokens(comments[i])]
    for i in scored:
        flags[i].polarity = polarity(comments[i])
    if len(scored) >= 8:
        xs = [midpoints[i] for i in scored]
        ys = [flags[i].polarity for i in scored]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        beta = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                if sxx else 0.0)
        alpha = my - beta * mx
        resid = {i: flags[i].polarity - (alpha + beta * midpoints[i])
                 for i in scored}
        med = statistics.median(resid.values())
        mad = statistics.median(abs(r - med) for r in resid.values()) or 1e-9
        for i, r in resid.items():
            flags[i].r6_scored = True
            if abs(r - med) > r6_mad_k * mad:
                flags[i].r6 = True
                direction = "more positive" if r - med > 0 else "more negative"
                flags[i].reasons.append(
                    f"R6: comment tone {direction} than usual for the mark "
                    f"(centred residual {r - med:+.3f}, > {r6_mad_k:g} MAD)")

    # ---- R7: long enough to pass R3, but very simple text ----
    for i, c in enumerate(comments):
        eligible = (r7_eligible[i] if r7_eligible is not None
                    else len((c or "").strip()) >= tau_chars)
        if not eligible:
            continue
        g = fk_grade(c)
        if g is not None and g < r7_grade:
            flags[i].r7 = True
            flags[i].reasons.append(
                f"R7: long comment but readability grade {g:.1f} < "
                f"{r7_grade:g}")

    # ---- R8: lexical drift against the same-band centroid ----
    vecs = _tfidf_matrix(comments)
    by_band: Dict[str, List[int]] = {}
    for i in range(n):
        band = honours_band(midpoints[i])
        if band and (comments[i] or "").strip():
            by_band.setdefault(band, []).append(i)
    sims: Dict[int, float] = {}
    for band, idxs in by_band.items():
        if len(idxs) < 5:
            continue                      # bands under five documents skipped
        centroid: Dict[str, float] = {}
        for i in idxs:                    # centroid includes the document
            for w, v in vecs[i].items():
                centroid[w] = centroid.get(w, 0.0) + v / len(idxs)
        for i in idxs:
            sims[i] = _cosine(vecs[i], centroid)
    if len(sims) >= 8:
        mean = statistics.fmean(sims.values())       # global, not per band
        sd = statistics.pstdev(sims.values()) or 1e-9
        for i, s in sims.items():
            flags[i].r8_scored = True
            if s < mean - r8_sd_k * sd:
                flags[i].r8 = True
                flags[i].reasons.append(
                    f"R8: wording unusual for the awarded band "
                    f"(similarity {s:.3f}, cohort mean {mean:.3f})")
    return flags
