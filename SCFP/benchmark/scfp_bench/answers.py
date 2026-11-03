"""Extract a final answer from model text and score it against the gold answer.

Models are asked to end with a line "Final answer: <answer>". When that line is
missing, extraction falls back to the last candidate in the text and the result
is flagged, so unparseable outputs can be counted and reported rather than
silently scored as wrong.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


@dataclass
class Extraction:
    value: Optional[str]     # normalized answer or None
    from_final_line: bool    # True if taken from an explicit "Final answer:" line


def _final_segment(text: str) -> Optional[str]:
    matches = FINAL_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def normalize_number(s: str) -> Optional[str]:
    nums = NUM_RE.findall(s.replace("$", "").replace("%", ""))
    if not nums:
        return None
    raw = nums[-1].replace(",", "").rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    if not math.isfinite(val):  # e.g. a runaway digit string that overflows float
        return None
    return str(int(val)) if val == int(val) else f"{val:.6g}"


def normalize_bool(s: str) -> Optional[str]:
    words = re.findall(r"\b(yes|no|true|false)\b", s.lower())
    if not words:
        return None
    return "yes" if words[-1] in ("yes", "true") else "no"


def normalize_choice(s: str, n_options: int = 4) -> Optional[str]:
    letters = "ABCDEFGH"[:n_options]
    m = re.findall(rf"\b([{letters}])\b", s.upper())
    return m[0] if m else None


def _boxed_segment(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (balanced braces), Qwen's default answer format."""
    i = (text or "").rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def extract(text: str, answer_type: str, n_options: int = 4, accept_boxed: bool = False) -> Extraction:
    """accept_boxed: also treat a final \\boxed{...} as an explicit answer when there is no
    "Final answer:" line (off by default, so SCFP v1 labels are unchanged)."""
    norm = {"number": normalize_number, "bool": normalize_bool,
            "choice": lambda s: normalize_choice(s, n_options)}[answer_type]
    seg = _final_segment(text)
    if seg is None and accept_boxed:
        seg = _boxed_segment(text)
    if seg is not None:
        val = norm(seg)
        if val is not None:
            return Extraction(val, True)
    # Fallback: last candidate anywhere in the text.
    if answer_type == "choice":
        m = re.findall(r"\b([A-H])\b", (text or "")[-200:])
        return Extraction(m[-1] if m else None, False)
    return Extraction(norm(text or ""), False)


def is_correct(extracted: Optional[str], gold: str, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "number":
        try:
            return abs(float(extracted) - float(gold)) < 1e-6
        except ValueError:
            return False
    return extracted.strip().lower() == gold.strip().lower()
