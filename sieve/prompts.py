"""The extraction prompt. Base and fine-tuned models get the same system prompt, so the comparison is fair."""
from __future__ import annotations

from typing import Any

from .schema import dumps

SYSTEM = """You extract job posts into one JSON object. Reply with the JSON only, no prose, no code fences.
Keys, in this order:
- title: the job title as written
- company: hiring company name, or null if not named
- location: "City, Country" as written, or null
- work_mode: "onsite" | "hybrid" | "remote" | null
- employment_type: "full_time" | "part_time" | "contract" | "internship" | null
- seniority: "intern" | "junior" | "mid" | "senior" | "lead"
- salary: null, or {"min": int or null, "max": int, "currency": ISO code, "period": "year" | "month" | "day" | "hour"} in whole currency units
- experience_years_min: integer or null
- skills: list of lower-case technology names mentioned"""

USER_TMPL = "Job post:\n<<<\n{text}\n>>>"


def messages(text: str, shots: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots or []:
        msgs.append({"role": "user", "content": USER_TMPL.format(text=s["text"])})
        msgs.append({"role": "assistant", "content": dumps(s["label"])})
    msgs.append({"role": "user", "content": USER_TMPL.format(text=text)})
    return msgs


def to_chat(text: str, label: dict[str, Any]) -> dict[str, Any]:
    return {"messages": messages(text) + [{"role": "assistant", "content": dumps(label)}]}
