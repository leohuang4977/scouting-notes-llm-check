"""The keyword list must be built from dev notes only.

Every top-level alternative in every pattern of code_keywords.py has to
match at least one dev note. A phrase that only test notes contain could
only have come from somewhere other than the dev notes, so this test
fails if one is added.
"""

import re

from src import code_keywords as K
from src.codebook import read_notes


def top_level_alternatives(pattern: str) -> list[str]:
    """Split a pattern on | outside parentheses, keeping a leading (?i) flag on each part."""
    flag = "(?i)" if pattern.startswith("(?i)") else ""
    body = pattern[len(flag):]
    parts, depth, current = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return [flag + p for p in parts]


def all_patterns() -> list[tuple[str, str]]:
    patterns = []
    for field, spec in K.FIELD_RULES.items():
        for name in ("topic", "skip"):
            if spec.get(name):
                patterns.append((f"{field} {name}", spec[name]))
        patterns += [(f"{field} -> {code}", p) for code, p in spec["rules"]]
    patterns += [("tone fallback favorable", K.FAVORABLE), ("tone fallback unfavorable", K.UNFAVORABLE)]
    return patterns


def test_every_keyword_alternative_matches_a_dev_note():
    notes = read_notes()
    dev_texts = notes.loc[notes["split"] == "dev", "text"].tolist()
    unmatched = []
    for where, pattern in all_patterns():
        for alternative in top_level_alternatives(pattern):
            if not any(re.search(alternative, text) for text in dev_texts):
                unmatched.append(f"{where}: {alternative}")
    assert unmatched == [], "keyword phrases not found in any dev note:\n" + "\n".join(unmatched)
