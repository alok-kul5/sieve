import json
import random
from pathlib import Path

from sieve.clean import dedup, mentioned_skills, reject_reason, salary_ok, split
from sieve.metrics import facts, parse, score
from sieve.prompts import messages, to_chat
from sieve.schema import JSON_SCHEMA, canonical, dumps
from sieve.synth import _indian, sample_salary, sample_spec, writer_prompt

FIX = Path(__file__).parent / "fixtures" / "mini_test.jsonl"
ROWS = [json.loads(line) for line in FIX.open(encoding="utf-8")]


def test_specs_are_deterministic_and_valid():
    from jsonschema import validate
    a = [sample_spec(random.Random(1)).label for _ in range(1)]
    b = [sample_spec(random.Random(1)).label for _ in range(1)]
    assert a == b
    rng = random.Random(0)
    for _ in range(300):
        spec = sample_spec(rng)
        validate(canonical(spec.label), JSON_SCHEMA)
        sal = spec.label["salary"]
        if sal and sal["min"] is not None:
            assert sal["min"] < sal["max"]
        assert spec.surfaces["title"] in writer_prompt(spec)


def test_indian_grouping():
    assert _indian(1800000) == "18,00,000"
    assert _indian(950000) == "9,50,000"
    assert _indian(500) == "500"


def test_salary_surface_matches_label():
    rng = random.Random(3)
    for _ in range(200):
        label, surf = sample_salary(rng, rng.choice(["USD", "INR", "EUR", "GBP"]), "full_time")
        assert salary_ok(f"Pay is {surf}. Apply now.", surf, label["period"])


def _raw(text, spec):
    return {"text": text, "surfaces": spec["surfaces"], "label": spec["label"]}


def test_filter_accepts_clean_and_rejects_drift():
    spec = {
        "label": {"title": "Senior Data Engineer", "company": "Kitefield", "location": "Austin, USA",
                  "work_mode": "remote", "employment_type": "full_time", "seniority": "senior",
                  "salary": {"min": 150000, "max": 180000, "currency": "USD", "period": "year"},
                  "experience_years_min": 5, "skills": ["airflow", "python"]},
        "surfaces": {"title": "Senior Data Engineer", "company": "Kitefield", "location": "Austin, USA",
                     "work_mode": "fully remote", "employment_type": "full-time", "salary": "$150k - $180k per year",
                     "experience": "5+ years", "skills": ["Python", "Airflow"]},
    }
    good = ("Kitefield is hiring a Senior Data Engineer! The role is fully remote (team based in Austin, USA), "
            "full-time, and pays $150k - $180k per year. You bring 5+ years building pipelines in Python and "
            "Airflow, love clean data, and enjoy mentoring others on the team. DM me to chat.")
    assert reject_reason(_raw(good, spec)) is None
    assert reject_reason(_raw(good.replace("Airflow", "Airflow and Docker"), spec)) == "stray:skills"
    assert reject_reason(_raw(good.replace("$150k - $180k per year", "great pay"), spec)) == "missing:salary"
    assert reject_reason(_raw(good.replace("fully remote", "hybrid"), spec)) == "mismatch:work_mode"
    assert reject_reason(_raw(good + " Ten years of history!", spec)) == "missing:experience"  # 10 != 5
    assert reject_reason(_raw(good.replace("5+ years", "five+ years"), spec)) is None  # paraphrase, same value


def test_skill_matching_uses_word_boundaries():
    assert mentioned_skills("We use Java and Kubernetes") == {"java", "kubernetes"}
    assert "java" not in mentioned_skills("Strong JavaScript skills")
    assert mentioned_skills("K8s + React.js") == {"kubernetes", "react"}


def test_dedup_and_split():
    rows = [{"text": r["text"]} for r in ROWS] + [{"text": ROWS[0]["text"] + " "}]
    kept, dropped = dedup(rows)
    assert dropped == 1 and len(kept) == len(ROWS)
    tr, va, te = split(list(range(100)))
    assert (len(tr), len(va), len(te)) == (70, 10, 20)
    assert not set(tr) & set(te)


def test_chat_format():
    r = ROWS[0]
    chat = to_chat(r["text"], r["label"])
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
    assert json.loads(chat["messages"][-1]["content"]) == canonical(r["label"])
    assert len(messages("x", ROWS[:2])) == 6


def test_parse_handles_fences_and_garbage():
    rec = dumps(ROWS[0]["label"])
    assert parse(rec)[1:] == (True, True)
    assert parse(f"```json\n{rec}\n```")[1:] == (True, True)
    assert parse("no json here") == (None, False, False)
    assert parse('{"title": "x"}')[1:] == (True, False)


def test_score_perfect_and_partial():
    golds = [r["label"] for r in ROWS]
    perfect = score([dumps(g) for g in golds], golds)
    assert perfect["exact_match"] == 1.0 and perfect["fact_f1"] == 1.0 and perfect["schema_valid"] == 1.0
    broken = [dumps(g) for g in golds]
    broken[0] = "oops"
    s = score(broken, golds)
    assert s["json_valid"] == (len(golds) - 1) / len(golds)
    assert 0 < s["fact_f1"] < 1
    assert ("salary.currency", "inr") in facts(golds[0])
