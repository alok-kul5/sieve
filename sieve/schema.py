"""The target record: one job post -> one strict JSON object."""
from __future__ import annotations

import json
from typing import Any

WORK_MODES = ["onsite", "hybrid", "remote"]
EMPLOYMENT_TYPES = ["full_time", "part_time", "contract", "internship"]
SENIORITY = ["intern", "junior", "mid", "senior", "lead"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "AUD", "CAD", "SGD"]
PERIODS = ["year", "month", "day", "hour"]

FIELDS = [
    "title",
    "company",
    "location",
    "work_mode",
    "employment_type",
    "seniority",
    "salary",
    "experience_years_min",
    "skills",
]

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": FIELDS,
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "company": {"type": ["string", "null"]},
        "location": {"type": ["string", "null"]},
        "work_mode": {"enum": WORK_MODES + [None]},
        "employment_type": {"enum": EMPLOYMENT_TYPES + [None]},
        "seniority": {"enum": SENIORITY},
        "salary": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["min", "max", "currency", "period"],
                    "properties": {
                        "min": {"type": ["integer", "null"]},
                        "max": {"type": "integer"},
                        "currency": {"enum": CURRENCIES},
                        "period": {"enum": PERIODS},
                    },
                },
            ]
        },
        "experience_years_min": {"type": ["integer", "null"], "minimum": 0},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
}


def canonical(record: dict[str, Any]) -> dict[str, Any]:
    """Fixed key order, sorted unique lower-case skills."""
    out = {k: record.get(k) for k in FIELDS}
    out["skills"] = sorted({str(s).strip().lower() for s in (record.get("skills") or [])})
    sal = out["salary"]
    if isinstance(sal, dict):
        out["salary"] = {k: sal.get(k) for k in ("min", "max", "currency", "period")}
    return out


def dumps(record: dict[str, Any]) -> str:
    """The exact string the model is trained to emit."""
    return json.dumps(canonical(record), ensure_ascii=False, separators=(", ", ": "))
