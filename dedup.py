"""Stitch overlapping Whisper window outputs into a single transcript."""
import re

_NORM_RE = re.compile(r"[^\w]+", re.UNICODE)


def _norm(word):
    return _NORM_RE.sub("", word.lower())


def merge_with_overlap(prev_text, new_text, max_overlap_words=25, min_overlap_words=2):
    """Append new_text to prev_text, removing the longest matching word
    sequence between the tail of prev_text and the head of new_text.
    """
    if not prev_text:
        return new_text.strip()
    if not new_text or not new_text.strip():
        return prev_text

    prev_words = prev_text.split()
    new_words = new_text.split()
    if not prev_words or not new_words:
        return (prev_text + " " + new_text).strip()

    prev_norm = [_norm(w) for w in prev_words]
    new_norm = [_norm(w) for w in new_words]

    max_check = min(len(prev_words), len(new_words), max_overlap_words)
    best = 0
    for n in range(max_check, min_overlap_words - 1, -1):
        if prev_norm[-n:] == new_norm[:n] and any(prev_norm[-n:]):
            best = n
            break

    merged = prev_words + new_words[best:]
    return " ".join(merged)
