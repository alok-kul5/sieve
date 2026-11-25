# Dataset card: Sieve job posts

Synthetic, messy job posts paired with a strict JSON record. Made for training and testing small models on
structured extraction. Files: `data/train.jsonl` (700), `data/val.jsonl` (99), `data/test.jsonl` (199),
plus `data/train_chat.jsonl` (the same train rows in chat format) and `data/stats.json`.

## How it was made

The label comes first and the text second, so the label can never drift from the text.

1. **Sample the truth** (`sieve/synth.py`). A seeded sampler picks a role family, a seniority marker, a title,
   a company name built from nonsense parts (no real companies), a city, work mode, employment type, a salary in
   the city's currency, years of experience and 2 to 6 skills from that role's pool. It also renders the
   surface strings a human would write: `₹18-24 LPA`, `€55.000 – €65.000 per year`, `9,00,000 INR` (Indian digit
   grouping), `$45-60/hr`, `up to S$9,000 monthly`, `five+ years`, `K8s`, `Postgres`, and so on.
2. **Write the post** (`sieve/generate.py`). A local model, **Phi-4-mini-instruct** (MIT licence, 4-bit, run
   in-process with transformers), wraps those facts in one of 8 styles: LinkedIn hype, recruiter email, Slack
   message, formal JD, WhatsApp forward, "who is hiring" comment, staffing-agency ad, tweet thread. The prompt
   names the facts to include, the facts to leave out (for null fields), and bans any other tools or figures.
   The writer is a different model family from the one being fine-tuned.
3. **Filter** (`sieve/clean.py`). Every post is checked against its label, and rejected if:
   - the title, company or location is not there verbatim;
   - the salary amount is not there (whitespace aside), or the pay period it signals is wrong or mixed;
   - any "N years" mention disagrees with `experience_years_min`, or years appear when the label is null;
   - the work mode or employment type it signals is missing or contradicts the label;
   - a technology from the 52-skill vocabulary is missing, or one not in the label appears;
   - money words appear when salary is null, or it is under 25 or over 260 words.
4. **Dedup**: exact hash on normalised text, then near-duplicates by 5-gram Jaccard > 0.6.
5. **Split**: seeded shuffle, 70 / 10 / 20 train / val / test.

| Stage | Rows |
|---|---:|
| Generated | 1,200 |
| Rejected by the label check | 202 (16.8%) |
| Near-duplicates removed | 0 |
| **Kept** | **998** |

Top reject reasons: salary amount changed (53), work mode contradicted (33), location rewritten (24),
length (20), title rewritten (17), skills dropped (14).

## Label conventions (learned by the fine-tune, stated in the prompt)

- `seniority` comes from the title marker: Intern → `intern`; Junior, Jr., Graduate, Entry-Level, Associate →
  `junior`; Senior, Sr. → `senior`; Lead, Principal, Staff → `lead`; **no marker → `mid`**.
- `salary` is in whole currency units: `₹18-24 LPA` → `{"min": 1800000, "max": 2400000, "currency": "INR",
  "period": "year"}`; `€55.000` → `55000`; `up to $150k` → `min: null`.
- `skills` are canonical, lower-case and sorted: `K8s` → `kubernetes`, `Postgres` → `postgresql`,
  `sklearn` → `scikit-learn`. A title like "Flutter Developer" counts as naming `flutter`.
- Facts that are not stated are `null`, never guessed.

## Train split at a glance

| | |
|---|---|
| Styles | whatsapp 105, tweet 100, agency 90, linkedin 89, formal 85, recruiter email 82, slack 75, reddit 74 |
| Seniority | mid 224, senior 186, junior 134, lead 110, intern 46 |
| Has salary | 512 of 700 (USD 138, INR 117, EUR 114, GBP 52, AUD 34, CAD 34, SGD 23) |
| Company not named | 146 |
| Mean length | 82 words |

## Limits

- Synthetic text from one writer model. Real posts are longer, mix several roles, and contain HTML junk.
- The skill vocabulary is closed (52 entries). A skill outside it is invisible to the label check.
- The filter guarantees the labelled facts are present and that no contradicting facts are present for the
  checked fields. It cannot catch every subtle paraphrase.

## Reproduce

```powershell
python -m sieve.generate --n 1200 --writer models/Phi-4-mini-instruct
python -m sieve.clean --raw data/raw.jsonl --out data
```
