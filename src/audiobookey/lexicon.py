"""Lexicon: pattern -> replacement text substitutions applied per page.

Format (one rule per line, '#' starts a comment):

    doughnut => donut
    r:\\bCo\\s?\\d+\\b => Company

Lines whose pattern starts with "r:" are Python regular expressions; all
other patterns are literal strings. The first ` -> ` / `=>` occurrence
separates pattern from replacement. Rules apply in file order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Supported separators (first match wins, left to right on the line).
_SEPARATORS = ("=>", "->")


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class LexiconRule:
    pattern: str
    replacement: str
    is_regex: bool
    line_number: int


class Lexicon:
    def __init__(self, rules: list[LexiconRule]) -> None:
        self.rules = rules

    def apply(self, text: str) -> str:
        for rule in self.rules:
            if rule.is_regex:
                # Escaping keeps the replacement literal (no \n/\d backref
                # semantics from re.sub's template syntax).
                text = re.sub(
                    rule.pattern, rule.replacement.replace("\\", "\\\\"), text
                )
            else:
                text = text.replace(rule.pattern, rule.replacement)
        return text


def _split_line(line: str, line_number: int) -> tuple[str, str]:
    for sep in _SEPARATORS:
        idx = line.find(sep)
        if idx != -1:
            pattern = line[:idx].strip()
            replacement = line[idx + len(sep) :].strip()
            if pattern:
                return pattern, replacement
    raise LexiconError(
        f"lexicon line {line_number}: missing separator "
        "(expected 'pattern => replacement')"
    )


def load_lexicon(path: str | Path) -> Lexicon:
    rules: list[LexiconRule] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, replacement = _split_line(line, line_number)
        is_regex = pattern.startswith("r:")
        if is_regex:
            pattern = pattern[2:].strip()
            try:
                re.compile(pattern)
            except re.error as exc:
                raise LexiconError(
                    f"lexicon line {line_number}: invalid regex: {exc}"
                ) from exc
        rules.append(
            LexiconRule(
                pattern=pattern,
                replacement=replacement,
                is_regex=is_regex,
                line_number=line_number,
            )
        )
    return Lexicon(rules)
