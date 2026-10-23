"""High-level search engine: options, overlap policies, highlighting, redaction."""

from __future__ import annotations

from .aho_corasick import AhoCorasick, Match


def resolve_overlaps(matches: list[Match], policy: str = "all") -> list[Match]:
    """Filter matches by overlap policy.

    * ``all``      – keep everything.
    * ``leftmost_longest`` – scan left to right, take the longest match at the
      leftmost start, then continue after its end (non-overlapping).
    * ``leftmost_first``   – like above but prefer the earliest-listed pattern.
    * ``max_cover``        – weighted interval scheduling maximising covered
      characters (non-overlapping).
    """
    ms = sorted(matches, key=lambda m: (m.start, -(m.end - m.start), m.pattern_id))
    if policy == "all":
        return sorted(matches)
    if policy in ("leftmost_longest", "leftmost_first"):
        if policy == "leftmost_first":
            ms = sorted(matches, key=lambda m: (m.start, m.pattern_id))
        out: list[Match] = []
        pos = 0
        for m in ms:
            if m.start >= pos:
                out.append(m)
                pos = m.end
        return out
    if policy == "max_cover":
        import bisect
        by_end = sorted(matches, key=lambda m: (m.end, m.start, m.pattern_id))
        ends = [m.end for m in by_end]
        best = [0] * (len(by_end) + 1)
        take = [False] * (len(by_end) + 1)
        prev = [0] * (len(by_end) + 1)
        for i, m in enumerate(by_end, 1):
            j = bisect.bisect_right(ends, m.start, 0, i - 1)
            with_m = best[j] + (m.end - m.start)
            if with_m > best[i - 1]:
                best[i], take[i], prev[i] = with_m, True, j
            else:
                best[i] = best[i - 1]
        res: list[Match] = []
        i = len(by_end)
        while i > 0:
            if take[i]:
                res.append(by_end[i - 1])
                i = prev[i]
            else:
                i -= 1
        return sorted(res)
    raise ValueError(f"unknown policy {policy!r}")


class SearchEngine:
    """Dictionary search over text with case folding and whole-word filtering."""

    def __init__(self, patterns: list[str], case_sensitive: bool = True,
                 whole_word: bool = False) -> None:
        self.original = list(dict.fromkeys(p for p in patterns if p))
        self.case_sensitive = case_sensitive
        self.whole_word = whole_word
        folded = self.original if case_sensitive else [p.lower() for p in self.original]
        self.ac = AhoCorasick(folded)

    @staticmethod
    def _is_word(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def search(self, text: str, policy: str = "all") -> list[Match]:
        hay = text if self.case_sensitive else text.lower()
        if len(hay) != len(text):  # case folding changed the length; fall back
            hay = "".join(c.lower() if len(c.lower()) == 1 else c for c in text)
        found = []
        for m in self.ac.iter_matches(hay):
            if self.whole_word:
                if m.start > 0 and self._is_word(text[m.start - 1]) and self._is_word(text[m.start]):
                    continue
                if m.end < len(text) and self._is_word(text[m.end]) and self._is_word(text[m.end - 1]):
                    continue
            found.append(Match(m.start, m.end, m.pattern_id, self.original[m.pattern_id]))
        return resolve_overlaps(found, policy)

    def highlight(self, text: str, left: str = "[", right: str = "]") -> str:
        """Wrap non-overlapping leftmost-longest matches with markers."""
        out, pos = [], 0
        for m in self.search(text, "leftmost_longest"):
            out.append(text[pos:m.start] + left + text[m.start:m.end] + right)
            pos = m.end
        out.append(text[pos:])
        return "".join(out)

    def redact(self, text: str, mask: str = "*") -> str:
        """Replace every matched character (union of all matches) by ``mask``."""
        covered = [False] * len(text)
        for m in self.search(text):
            for i in range(m.start, m.end):
                covered[i] = True
        return "".join(mask if c else ch for ch, c in zip(text, covered))

    def replace(self, text: str, mapping: dict[str, str]) -> str:
        """Replace leftmost-longest matches using ``mapping`` (missing keys kept)."""
        out, pos = [], 0
        for m in self.search(text, "leftmost_longest"):
            out.append(text[pos:m.start])
            out.append(mapping.get(m.pattern, text[m.start:m.end]))
            pos = m.end
        out.append(text[pos:])
        return "".join(out)
