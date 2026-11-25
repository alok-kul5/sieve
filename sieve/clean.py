"""Filter, dedup and split raw synthetic posts into train/val/test chat files.

    python -m sieve.clean --raw data/raw.jsonl --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .prompts import to_chat
from .schema import canonical
from .synth import SKILLS

DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
MONEY_RE = re.compile(r"[$€£₹]|\b(?:USD|EUR|GBP|INR|AUD|CAD|SGD|LPA|lakhs?|per annum|salary|stipend)\b|\d+\s?k\b",
                      re.IGNORECASE)
YEARS_RE = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)\b",
                      re.IGNORECASE)
WORD_NUM = {w: str(i) for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                                "nine", "ten"])}
MODE_RE =re.compile(r"\b(remote|hybrid|on-?site|in-office|office-based|wfh|work from (?:home|anywhere))\b", re.IGNORECASE)
EMPLOY_RE = re.compile(r"\b(full[- ]?time|part[- ]?time|contract|freelance|permanent|internship|fte)\b", re.IGNORECASE)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(DASHES)
    return re.sub(r"\s+", " ", t).strip().lower()


def contains(text: str, needle: str) -> bool:
    return norm(needle) in norm(text)


def _surface_re(form: str) -> re.Pattern:
    return re.compile(r"(?<![\w.+#])" + re.escape(form.lower()) + r"(?![\w+#])")


SKILL_PATTERNS = {canon: [_surface_re(f) for f in forms + [canon]] for canon, forms in SKILLS.items()}


def mentioned_skills(text: str) -> set[str]:
    t = norm(text)
    return {c for c, pats in SKILL_PATTERNS.items() if any(p.search(t) for p in pats)}


MODE_CUES = {
    "remote": re.compile(r"\b(remote|wfh|work from (?:home|anywhere)|home-based)\b", re.IGNORECASE),
    "hybrid": re.compile(r"\bhybrid\b", re.IGNORECASE),
    "onsite": re.compile(r"\b(on-?site|in-office|in office|office-based|office based)\b", re.IGNORECASE),
}
EMPLOY_CUES = {
    "full_time": re.compile(r"\b(full[- ]?time|permanent|fte)\b", re.IGNORECASE),
    "part_time": re.compile(r"\bpart[- ]?time\b", re.IGNORECASE),
    "contract": re.compile(r"\b(contract|freelance|contractor)\b", re.IGNORECASE),
    "internship": re.compile(r"\b(intern|internship)\b", re.IGNORECASE),
}


PERIOD_CUES = {
    "year": re.compile(r"(\bper year|\bper annum|/ ?yr\b|/ ?year\b|\ba year\b|\bannual(?:ly)?\b|\bp\.a\.|\bpa\b|\blpa\b"
                       r"|\blakhs? per annum|\blakh ?/ ?yr\b|\byearly\b)", re.IGNORECASE),
    "month": re.compile(r"(\bper month\b|/ ?month\b|/ ?mo\b|\ba month\b|\bmonthly\b)", re.IGNORECASE),
    "day": re.compile(r"(\bper day\b|/ ?day\b|\bday rate\b|\bdaily\b)", re.IGNORECASE),
    "hour": re.compile(r"(\bper hour\b|/ ?hr\b|/ ?hour\b|\ban hour\b|\bhourly\b)", re.IGNORECASE),
}
_PERIOD_TAIL = re.compile(r"\s*(" + "|".join(p.pattern for p in PERIOD_CUES.values()) + r")\s*$", re.IGNORECASE)


def salary_ok(text: str, surface: str, period: str) -> bool:
    """Amount part verbatim (ignoring spaces), period signalled anywhere."""
    core = _PERIOD_TAIL.sub("", norm(surface))
    squash = lambda x: re.sub(r"\s+", "", norm(x))
    if squash(core) not in squash(text):
        return False
    hits = {k for k, p in PERIOD_CUES.items() if p.search(norm(text))}
    return period in hits and not (hits - {period})


def cue_ok(text: str, want: str, cues: dict[str, re.Pattern], allow: dict[str, set[str]]) -> bool:
    """`want` must be signalled; other categories may appear only if allowed (e.g. 'remote' days in a hybrid post)."""
    hits = {k for k, p in cues.items() if p.search(text)}
    return want in hits and not (hits - {want} - allow.get(want, set()))


def reject_reason(rec: dict[str, Any]) -> str | None:
    """Return why a raw post fails the label check, or None if it is clean."""
    text, s, lab = rec.get("text") or "", rec["surfaces"], rec["label"]
    words = len(text.split())
    if words < 25 or words > 260:
        return "length"
    # facts whose exact value matters must appear verbatim
    for key in ("title", "company", "location"):
        if s.get(key) and not contains(text, s[key]):
            return f"missing:{key}"
    # every "N years" mention must agree with the label; at least one if the label has a value
    years = [WORD_NUM.get(m.group(1).lower(), m.group(1)) for m in YEARS_RE.finditer(norm(text))]
    want_years = lab["experience_years_min"]
    if want_years is None and years:
        return "stray:experience"
    if want_years is not None and (not years or any(int(y) != want_years for y in years)):
        return "missing:experience"
    if s.get("salary") and not salary_ok(text, s["salary"], lab["salary"]["period"]):
        return "missing:salary"
    nt = norm(text)
    # categorical facts may be paraphrased, but must signal the labelled category and nothing contradictory
    if lab["work_mode"] and not cue_ok(nt, lab["work_mode"], MODE_CUES, {"hybrid": {"remote", "onsite"}}):
        return "mismatch:work_mode"
    title_intern = lab["seniority"] == "intern"
    if lab["employment_type"] and not cue_ok(nt, lab["employment_type"], EMPLOY_CUES,
                                             {"internship": {"full_time", "part_time"}}):
        return "mismatch:employment_type"
    t = nt
    for key in ("title", "company", "location"):
        if s.get(key):  # blank out required strings before checking for stray facts
            t = t.replace(norm(s[key]), " ")
    if not s.get("salary") and MONEY_RE.search(t):
        return "stray:salary"
    if not lab["work_mode"] and MODE_RE.search(t):
        return "stray:work_mode"
    if not lab["employment_type"] and not title_intern and EMPLOY_RE.search(t):
        return "stray:employment_type"
    found = mentioned_skills(text)
    want = set(lab["skills"])
    if want - found:
        return "missing:skills"
    if found - want:
        return "stray:skills"
    return None


def shingles(text: str, n: int = 5) -> set[str]:
    w = re.findall(r"\w+", norm(text))
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def dedup(recs: list[dict], thresh: float = 0.6) -> tuple[list[dict], int]:
    """Exact-hash dedup, then near-duplicate removal by 5-gram Jaccard."""
    seen, kept, sigs, dropped = set(), [], [], 0
    for r in recs:
        h = hashlib.sha1(norm(r["text"]).encode()).hexdigest()
        if h in seen:
            dropped += 1
            continue
        sh = shingles(r["text"])
        if any(len(sh & o) / max(1, len(sh | o)) > thresh for o in sigs):
            dropped += 1
            continue
        seen.add(h)
        sigs.append(sh)
        kept.append(r)
    return kept, dropped


def split(recs: list[dict], seed: int = 13, val: float = 0.1, test: float = 0.2):
    rng = random.Random(seed)
    recs = recs[:]
    rng.shuffle(recs)
    n_test, n_val = int(len(recs) * test), int(len(recs) * val)
    return recs[n_test + n_val:], recs[n_test:n_test + n_val], recs[:n_test]


def to_example(r: dict) -> dict:
    return {"id": r["id"], "style": r["style"], "text": r["text"].strip(), "label": canonical(r["label"])}


def stats(rows: list[dict]) -> dict[str, Any]:
    lab = [r["label"] for r in rows]
    return {
        "n": len(rows),
        "style": dict(Counter(r["style"] for r in rows).most_common()),
        "seniority": dict(Counter(x["seniority"] for x in lab).most_common()),
        "has_salary": sum(x["salary"] is not None for x in lab),
        "currency": dict(Counter(x["salary"]["currency"] for x in lab if x["salary"]).most_common()),
        "null_company": sum(x["company"] is None for x in lab),
        "mean_words": round(sum(len(r["text"].split()) for r in rows) / max(1, len(rows)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw.jsonl")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    raw = [json.loads(line) for line in Path(args.raw).open(encoding="utf-8")]
    reasons = Counter()
    clean = []
    for r in raw:
        why = reject_reason(r)
        reasons[why or "kept"] += 1
        if not why:
            clean.append(r)
    clean, dups = dedup(clean)
    train, val, test = split([to_example(r) for r in clean])

    out = Path(args.out)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "train_chat.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(to_chat(r["text"], r["label"]), ensure_ascii=False) + "\n")

    report = {"raw": len(raw), "filter": dict(reasons.most_common()), "near_duplicates_dropped": dups,
              "train": stats(train), "val": stats(val), "test": stats(test)}
    (out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw", "filter", "near_duplicates_dropped")}, indent=2))
    print({k: report[k]["n"] for k in ("train", "val", "test")})


if __name__ == "__main__":
    main()
