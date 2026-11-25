"""Scoring: JSON validity, exact match, per-field accuracy and fact-level micro F1."""
from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from .schema import FIELDS, JSON_SCHEMA, canonical

_VALIDATOR = Draft202012Validator(JSON_SCHEMA)


def parse(output: str) -> tuple[dict | None, bool, bool]:
    """Return (record or None, parsed_as_json, schema_valid).

    Strict: the reply must be one JSON object. We tolerate surrounding whitespace and a
    markdown fence (common for prompted models) when parsing, but report it via `strict`.
    """
    s = output.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j < i:
            return None, False, False
        s = s[i:j + 1]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None, False, False
    if not isinstance(obj, dict):
        return None, False, False
    return obj, True, _VALIDATOR.is_valid(obj)


def _norm_str(v: Any) -> Any:
    return re.sub(r"\s+", " ", v.strip().lower()) if isinstance(v, str) else v


def facts(rec: dict[str, Any]) -> set[tuple]:
    """A record as a set of atomic (field, value) facts; nulls are not facts."""
    out: set[tuple] = set()
    for k in FIELDS:
        v = rec.get(k)
        if k == "skills":
            for s in v or []:
                out.add(("skill", _norm_str(str(s))))
        elif k == "salary":
            if isinstance(v, dict):
                for sk in ("min", "max", "currency", "period"):
                    if v.get(sk) is not None:
                        out.add((f"salary.{sk}", _norm_str(v.get(sk))))
        elif v is not None:
            out.add((k, _norm_str(v)))
    return out


def field_correct(pred: dict | None, gold: dict) -> dict[str, bool]:
    if pred is None:
        return {k: False for k in FIELDS}
    p = canonical({k: pred.get(k) for k in FIELDS}) if isinstance(pred.get("skills", []), list) else pred
    res = {}
    for k in FIELDS:
        a, b = p.get(k), gold.get(k)
        if k == "salary" and isinstance(a, dict) and isinstance(b, dict):
            res[k] = all(_norm_str(a.get(x)) == _norm_str(b.get(x)) for x in ("min", "max", "currency", "period"))
        elif k == "skills":
            res[k] = sorted(map(_norm_str, a or [])) == sorted(map(_norm_str, b or []))
        else:
            res[k] = _norm_str(a) == _norm_str(b)
    return res


def score(outputs: list[str], golds: list[dict]) -> dict[str, Any]:
    n = len(golds)
    valid_json = valid_schema = exact = 0
    tp = fp = fn = 0
    per_field = {k: 0 for k in FIELDS}
    for out, gold in zip(outputs, golds):
        pred, ok_json, ok_schema = parse(out)
        valid_json += ok_json
        valid_schema += ok_schema
        fc = field_correct(pred if ok_json else None, gold)
        for k, v in fc.items():
            per_field[k] += v
        exact += all(fc.values())
        pf, gf = (facts(pred) if pred else set()), facts(gold)
        tp += len(pf & gf)
        fp += len(pf - gf)
        fn += len(gf - pf)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "json_valid": valid_json / n,
        "schema_valid": valid_schema / n,
        "exact_match": exact / n,
        "fact_precision": prec,
        "fact_recall": rec,
        "fact_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "field_accuracy": {k: v / n for k, v in per_field.items()},
    }
