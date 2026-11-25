"""Template 'writer' so CI can run the data pipeline without any model."""
import argparse
import json
import random
from dataclasses import asdict

from sieve.synth import sample_spec

FILLER = ["Great team, great culture.", "Apply via the link below.", "Benefits include learning budgets and health cover.",
          "We ship fast and care about quality.", "Interviews are quick and friendly."]


def render(spec, rng) -> str:
    s = spec.surfaces
    parts = [f"We are hiring a {s['title']}" + (f" at {s['company']}" if s["company"] else "") + "."]
    if s["location"]:
        parts.append(f"Based in {s['location']}.")
    if s["work_mode"]:
        parts.append(f"This role is {s['work_mode']}.")
    if s["employment_type"]:
        parts.append(f"It is {s['employment_type']}.")
    if s["salary"]:
        parts.append(f"Compensation: {s['salary']}.")
    if s["experience"]:
        parts.append(f"You bring {s['experience']} in the field.")
    parts.append("Stack: " + ", ".join(s["skills"]) + ".")
    parts += rng.sample(FILLER, 3)
    return " ".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            spec = sample_spec(rng)
            rec = {"id": f"mock-{i:04d}", "writer": "template", "style": spec.style, "text": render(spec, rng),
                   **{k: v for k, v in asdict(spec).items() if k != "style"}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
