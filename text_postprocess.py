"""Text post-processing: hallucination filter + voice punctuation commands."""
import re


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------
_HALLUCINATIONS_RAW = [
    "thank you for watching", "thanks for watching", "thank you for watching!",
    "please subscribe", "please like and subscribe", "like and subscribe",
    "subtitles by the amara.org community", "subtitles by amara.org",
    "you", "thank you", "music", "[music]", "(music)",
    "[applause]", "(applause)", "[silence]", "bye", "bye bye",
]


def _normalize(t):
    if not t:
        return ""
    t = t.strip().lower()
    while t and t[-1] in ".!?,;:- ":
        t = t[:-1]
    return t.strip()


_HALLUCINATIONS = frozenset(_normalize(t) for t in _HALLUCINATIONS_RAW)


def is_hallucination(text):
    if not text or not text.strip():
        return True
    return _normalize(text) in _HALLUCINATIONS


# ---------------------------------------------------------------------------
# Voice punctuation
# ---------------------------------------------------------------------------
# Each pattern matches the spoken keyword whether it appears mid-sentence
# (after a non-whitespace char) or at the very start of the text.
# The old (?<=\S)\s+ lookbehind alone broke "new line" at the start of a
# transcription. The |\A branch handles that case cleanly.
def _kw(*words, trail=""):
    body = r"(?:" + r"\s+".join(words) + r")"
    return re.compile(
        r"(?:(?<=\S)\s+|\A\s*)" + body + trail,
        re.IGNORECASE,
    )


_PUNCT_REPLACEMENTS = [
    # ── newlines ─────────────────────────────────────────────────────
    (_kw("new", "paragraph", trail=r"\s*\.?"),                    "\n\n"),
    (_kw("new", "line",      trail=r"\s*\.?"),                    "\n"),
    (_kw("next", "line",     trail=r"\s*\.?"),                    "\n"),
    (_kw("newline",          trail=r"\s*\.?"),                    "\n"),
    # ── sentence-ending (trailing space so next word isn't butted up) ─
    (_kw("full", "stop",           trail=r"\s*\.?"),              ". "),
    (_kw("period",                 trail=r"\s*\.?"),              ". "),
    (_kw("question", "mark",       trail=r"\s*[?.]?"),             "? "),
    (_kw("exclamation", "(?:mark|point)", trail=r"\s*[!.]?"),     "! "),
    # ── inline ───────────────────────────────────────────────────────
    (_kw("comma",                  trail=r"\s*,?"),               ", "),
    (_kw("semi-?colon",            trail=r"\s*;?"),               "; "),
    (_kw("colon",                  trail=r"\s*:?"),               ": "),
    # ── quotes (trail=\s* consumes the space after the keyword so the
    #           quoted word butts directly against the quote mark) ──
    (_kw("open",  r"quot(?:e|es|ation)", trail=r"\s*"),           ' "'),
    (_kw("close", r"quot(?:e|es|ation)", trail=r"\s*"),           '" '),
    # ── dashes ───────────────────────────────────────────────────────
    (_kw("em", "dash",             trail=""),                     " — "),
    (_kw("en", "dash",             trail=""),                     " – "),
    (_kw("hyphen",                 trail=""),                     "-"),
    # ── brackets ─────────────────────────────────────────────────────
    (_kw("open",  r"(?:paren|parenthesis|bracket)", trail=""),    " ("),
    (_kw("close", r"(?:paren|parenthesis|bracket)", trail=""),    ") "),
    # ── ellipsis ─────────────────────────────────────────────────────
    (_kw(r"(?:dot\s+dot\s+dot|ellipsis)", trail=""),              "... "),
]


def apply_punctuation(text):
    """Convert spoken punctuation words to actual punctuation characters."""
    if not text:
        return text
    for pattern, replacement in _PUNCT_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    # Collapse any double-spaces introduced by replacements.
    # Strip leading/trailing spaces only — not newlines, since the user may
    # have started with "new line" and we must preserve that \n.
    text = re.sub(r"  +", " ", text)
    return text.strip(" \t")
