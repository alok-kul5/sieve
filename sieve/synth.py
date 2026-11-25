"""Sample a ground-truth record, then render the surface strings a writer model must use.

The label is fixed *before* any text exists, so it can never drift from the text:
the writer (a local Ollama model) only gets to wrap the facts in messy prose, and the
filter in clean.py rejects any post that drops a fact or adds one that is not in the label.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# canonical skill -> surface forms the writer may use (all must be safe to grep for)
SKILLS: dict[str, list[str]] = {
    "python": ["Python", "Python 3"],
    "java": ["Java"],
    "javascript": ["JavaScript"],
    "typescript": ["TypeScript"],
    "golang": ["Golang"],
    "rust": ["Rust"],
    "c++": ["C++"],
    "kotlin": ["Kotlin"],
    "swift": ["Swift"],
    "sql": ["SQL"],
    "react": ["React", "React.js", "ReactJS"],
    "next.js": ["Next.js", "NextJS"],
    "vue": ["Vue.js", "VueJS"],
    "node.js": ["Node.js", "NodeJS"],
    "django": ["Django"],
    "fastapi": ["FastAPI"],
    "spring boot": ["Spring Boot"],
    "graphql": ["GraphQL"],
    "postgresql": ["PostgreSQL", "Postgres"],
    "mysql": ["MySQL"],
    "mongodb": ["MongoDB", "Mongo DB"],
    "redis": ["Redis"],
    "kafka": ["Kafka", "Apache Kafka"],
    "spark": ["Apache Spark", "PySpark"],
    "airflow": ["Airflow"],
    "dbt": ["dbt"],
    "snowflake": ["Snowflake"],
    "aws": ["AWS", "Amazon Web Services"],
    "gcp": ["GCP", "Google Cloud"],
    "azure": ["Azure"],
    "docker": ["Docker"],
    "kubernetes": ["Kubernetes", "K8s"],
    "terraform": ["Terraform"],
    "ansible": ["Ansible"],
    "jenkins": ["Jenkins"],
    "github actions": ["GitHub Actions"],
    "linux": ["Linux"],
    "pytorch": ["PyTorch"],
    "tensorflow": ["TensorFlow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas", "Pandas"],
    "mlflow": ["MLflow"],
    "langchain": ["LangChain"],
    "tableau": ["Tableau"],
    "power bi": ["Power BI", "PowerBI"],
    "figma": ["Figma"],
    "selenium": ["Selenium"],
    "cypress": ["Cypress"],
    "splunk": ["Splunk"],
    "siem": ["SIEM"],
    "wireshark": ["Wireshark"],
    "flutter": ["Flutter"],
}

ROLES: dict[str, dict[str, Any]] = {
    "backend": {"titles": ["Backend Engineer", "Software Engineer", "Backend Developer", "API Engineer"],
                "skills": ["python", "java", "golang", "rust", "node.js", "django", "fastapi", "spring boot",
                           "postgresql", "mysql", "redis", "kafka", "docker", "kubernetes", "aws", "graphql", "sql"]},
    "frontend": {"titles": ["Frontend Engineer", "Frontend Developer", "UI Engineer", "Web Developer"],
                 "skills": ["javascript", "typescript", "react", "next.js", "vue", "graphql", "figma", "cypress",
                            "node.js"]},
    "fullstack": {"titles": ["Full Stack Engineer", "Full Stack Developer", "Product Engineer"],
                  "skills": ["javascript", "typescript", "react", "next.js", "node.js", "python", "django",
                             "postgresql", "mongodb", "docker", "aws", "graphql", "redis"]},
    "data": {"titles": ["Data Engineer", "Analytics Engineer", "Data Platform Engineer"],
             "skills": ["python", "sql", "spark", "airflow", "dbt", "snowflake", "kafka", "aws", "gcp",
                        "postgresql", "docker", "terraform"]},
    "analyst": {"titles": ["Data Analyst", "Business Intelligence Analyst", "Product Analyst"],
                "skills": ["sql", "python", "tableau", "power bi", "pandas", "snowflake", "dbt"]},
    "ml": {"titles": ["Machine Learning Engineer", "ML Engineer", "Data Scientist", "AI Engineer",
                      "Applied Scientist"],
           "skills": ["python", "pytorch", "tensorflow", "scikit-learn", "pandas", "mlflow", "langchain", "sql",
                      "docker", "kubernetes", "aws", "gcp", "spark", "fastapi"]},
    "devops": {"titles": ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer",
                          "Cloud Engineer"],
               "skills": ["aws", "gcp", "azure", "kubernetes", "docker", "terraform", "ansible", "jenkins",
                          "github actions", "linux", "python", "golang"]},
    "security": {"titles": ["Security Engineer", "SOC Analyst", "Cloud Security Engineer",
                            "Detection Engineer"],
                 "skills": ["splunk", "siem", "wireshark", "python", "aws", "azure", "linux", "terraform",
                            "kubernetes"]},
    "mobile": {"titles": ["Mobile Engineer", "Android Developer", "iOS Developer", "Flutter Developer"],
               "skills": ["kotlin", "swift", "flutter", "java", "typescript", "react", "graphql"]},
    "qa": {"titles": ["QA Engineer", "SDET", "Test Automation Engineer"],
           "skills": ["selenium", "cypress", "python", "java", "javascript", "jenkins", "github actions"]},
}

# seniority marker -> label. No marker means "mid" (house convention, learned not prompted).
SENIORITY_PREFIX = {
    "intern": ["Intern", "Summer Intern"],
    "junior": ["Junior", "Jr.", "Graduate", "Entry-Level", "Associate"],
    "mid": [""],
    "senior": ["Senior", "Sr."],
    "lead": ["Lead", "Principal", "Staff"],
}

LOCATIONS = [  # (location string, currency)
    ("Bengaluru, India", "INR"), ("Pune, India", "INR"), ("Hyderabad, India", "INR"), ("Mumbai, India", "INR"),
    ("Gurugram, India", "INR"), ("Chennai, India", "INR"),
    ("San Francisco, USA", "USD"), ("New York, USA", "USD"), ("Austin, USA", "USD"), ("Seattle, USA", "USD"),
    ("Chicago, USA", "USD"), ("Boston, USA", "USD"),
    ("London, UK", "GBP"), ("Manchester, UK", "GBP"), ("Edinburgh, UK", "GBP"),
    ("Berlin, Germany", "EUR"), ("Munich, Germany", "EUR"), ("Amsterdam, Netherlands", "EUR"),
    ("Dublin, Ireland", "EUR"), ("Paris, France", "EUR"), ("Lisbon, Portugal", "EUR"),
    ("Sydney, Australia", "AUD"), ("Melbourne, Australia", "AUD"),
    ("Toronto, Canada", "CAD"), ("Vancouver, Canada", "CAD"),
    ("Singapore, Singapore", "SGD"),
]

CO_A = ["North", "Bright", "Blue", "Quiet", "Iron", "Silver", "Cedar", "Lumen", "Arc", "Hollow", "Ember",
        "Kite", "Tidal", "Copper", "Orbit", "Maple", "Granite", "Velvet", "Zephyr", "Pixel", "Atlas", "Nimbus"]
CO_B = ["loop", "wind", "stack", "forge", "leaf", "works", "path", "beam", "grid", "nest", "field", "mint",
        "spring", "harbor", "line", "shift", "craft", "lane", "bay", "point"]
CO_C = ["", "", "", " Labs", " Technologies", " Analytics", " Health", " Pay", " Systems", " AI", " Logistics",
        " Robotics", " Games", " Energy", " Bank", " Software"]

WORK_MODE_SURFACE = {
    "remote": ["fully remote", "100% remote", "remote-first", "work from anywhere", "WFH (remote)"],
    "hybrid": ["hybrid (3 days in office)", "hybrid, 2 days WFH", "hybrid working", "hybrid model"],
    "onsite": ["on-site", "onsite, 5 days a week", "in-office role", "office-based"],
}
EMPLOYMENT_SURFACE = {
    "full_time": ["full-time", "Full Time", "permanent position", "FTE"],
    "part_time": ["part-time", "part time (20 hrs/week)"],
    "contract": ["contract", "6-month contract", "freelance contract", "12-month contract role"],
    "internship": ["internship", "paid internship"],
}
NUM_WORDS = {9: "nine", 10: "ten", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}

# yearly ranges per currency (low, high) in whole units
YEAR_RANGE = {"USD": (60_000, 260_000), "EUR": (40_000, 130_000), "GBP": (30_000, 120_000),
              "INR": (400_000, 6_000_000), "AUD": (70_000, 220_000), "CAD": (60_000, 200_000),
              "SGD": (50_000, 220_000)}
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "AUD": "A$", "CAD": "CA$", "SGD": "S$"}


def _round(x: float, step: int) -> int:
    return int(round(x / step) * step)


def _thousands(n: int, sep: str = ",") -> str:
    return f"{n:,}".replace(",", sep)


def _indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def sample_salary(rng: random.Random, currency: str, employment: str | None) -> tuple[dict | None, str | None]:
    """Return (label, surface string)."""
    sym = SYMBOL[currency]
    period = "year"
    r = rng.random()
    if employment == "contract" and r < 0.5:
        period = "day" if currency in ("GBP", "EUR") else "hour"
    elif employment in ("internship", "part_time") or (currency == "INR" and r < 0.15):
        period = "month" if currency == "INR" or rng.random() < 0.5 else "hour"
    elif r < 0.08:
        period = "hour"

    lo, hi = YEAR_RANGE[currency]
    if period == "year":
        a = rng.uniform(lo, hi * 0.8)
        step = 100_000 if currency == "INR" else 5_000
        mn = _round(a, step)
        mx = _round(a * rng.uniform(1.12, 1.45), step)
    elif period == "month":
        a = rng.uniform(lo, hi * 0.6) / 12
        step = 5_000 if currency == "INR" else 250
        mn = max(step, _round(a, step))
        mx = _round(mn * rng.uniform(1.1, 1.4), step)
    elif period == "day":
        mn = _round(rng.uniform(250, 650), 25)
        mx = _round(mn * rng.uniform(1.1, 1.35), 25)
    else:  # hour
        top = 35 if employment == "internship" else 110
        mn = _round(rng.uniform(15, top) * (80 if currency == "INR" else 1), 5)
        mx = _round(mn * rng.uniform(1.1, 1.5), 5)
    if mx <= mn:
        mx = mn + (step if period in ("year", "month") else 5)
    upto = rng.random() < 0.12
    label = {"min": None if upto else mn, "max": mx, "currency": currency, "period": period}

    per = {"year": rng.choice(["per year", "per annum", "/yr", "a year", "annually"]),
           "month": rng.choice(["per month", "/month", "a month", "monthly"]),
           "day": rng.choice(["per day", "/day", "day rate"]),
           "hour": rng.choice(["per hour", "/hr", "an hour", "hourly"])}[period]

    style = rng.random()  # one number style per post

    def fmt(n: int) -> str:
        if period == "year" and style < 0.45 and mn % 1000 == 0 and n % 1000 == 0:
            return f"{n // 1000}k"
        if currency == "EUR" and style < 0.7:
            return _thousands(n, ".")
        if currency == "INR":
            return _indian(n)
        return _thousands(n)

    lakhs = currency == "INR" and period == "year" and rng.random() < 0.6
    if lakhs:
        a, b = f"{mn / 100_000:g}", f"{mx / 100_000:g}"
        unit = rng.choice(["LPA", "lakhs per annum", "L p.a.", "lakh/yr"])
        surf = f"up to ₹{b} {unit}" if upto else rng.choice([f"₹{a}-{b} {unit}", f"{a} to {b} {unit}",
                                                             f"INR {a}–{b} {unit}"])
        return label, surf
    a, b = fmt(mn), fmt(mx)
    if upto:
        surf = rng.choice([f"up to {sym}{b} {per}", f"max {b} {currency} {per}"])
    else:
        surf = rng.choice([f"{sym}{a} - {sym}{b} {per}", f"{sym}{a}–{b} {per}", f"{currency} {a} to {b} {per}",
                           f"{a}-{b} {currency} {per}"])
    return label, surf


EXP_RANGE = {"junior": (0, 2), "mid": (2, 5), "senior": (4, 8), "lead": (6, 10)}


def sample_experience(rng: random.Random, sen: str) -> tuple[int | None, str | None]:
    if sen == "intern" or rng.random() < 0.2:
        return None, None
    n = rng.randint(*EXP_RANGE[sen])
    if n == 0:
        return 0, rng.choice(["0-2 years", "0+ years", "0-1 years"])
    forms = [f"{n}+ years", f"at least {n} years", f"minimum {n} yrs", f"{n}-{n + rng.randint(2, 3)} years",
             f"{NUM_WORDS[n]}+ years", f"{n}+ yrs of experience"]
    return n, rng.choice(forms)


def company_name(rng: random.Random) -> str:
    return rng.choice(CO_A) + rng.choice(CO_B) + rng.choice(CO_C)


STYLES = {
    "linkedin": "an over-excited LinkedIn hiring post with emojis and hashtags",
    "recruiter_email": "a cold recruiter email to a candidate, with a greeting and a sign-off from a recruiter",
    "slack": "a short, casual Slack message in a #jobs channel, lowercase, a bit sloppy",
    "formal": "a formal job description with a heading and bullet points for responsibilities and requirements",
    "whatsapp": "a forwarded WhatsApp message in a job group, with line breaks and a few typos",
    "reddit": "a comment in a monthly 'who is hiring' thread, using | separators on the first line",
    "agency": "a staffing-agency ad that talks about 'our client' and lists perks",
    "tweet": "a two-tweet thread (1/2, 2/2) announcing the opening",
}


@dataclass
class Spec:
    """Ground truth + the exact strings the writer has to use."""
    label: dict[str, Any]
    style: str
    surfaces: dict[str, Any] = field(default_factory=dict)  # field -> surface string(s), None if absent


def sample_spec(rng: random.Random) -> Spec:
    role = rng.choice(list(ROLES))
    r = ROLES[role]
    sen = rng.choices(list(SENIORITY_PREFIX), weights=[0.06, 0.2, 0.3, 0.3, 0.14])[0]
    prefix = rng.choice(SENIORITY_PREFIX[sen])
    base = rng.choice(r["titles"])
    if sen == "intern":
        title = f"{base} {prefix}" if rng.random() < 0.5 else f"{prefix} - {base}"
    else:
        title = f"{prefix} {base}".strip()

    employment = "internship" if sen == "intern" else rng.choices(
        ["full_time", "part_time", "contract", None], weights=[0.6, 0.07, 0.18, 0.15])[0]
    mode = rng.choices(["remote", "hybrid", "onsite", None], weights=[0.3, 0.3, 0.25, 0.15])[0]
    loc, currency = rng.choice(LOCATIONS)
    location = None if (mode == "remote" and rng.random() < 0.5) or rng.random() < 0.08 else loc

    style = rng.choice(list(STYLES))
    company = None if style == "agency" or rng.random() < 0.1 else company_name(rng)

    salary, salary_s = (None, None)
    if rng.random() < 0.75:
        salary, salary_s = sample_salary(rng, currency, employment)
    exp, exp_s = sample_experience(rng, sen)

    k = rng.randint(2, 6)
    skills = rng.sample(r["skills"], k=min(k, len(r["skills"])))
    skill_s = [rng.choice(SKILLS[s]) for s in skills]

    label = {
        "title": title,
        "company": company,
        "location": location,
        "work_mode": mode,
        "employment_type": employment,
        "seniority": sen,
        "salary": salary,
        "experience_years_min": exp,
        "skills": sorted(skills),
    }
    surfaces = {
        "title": title,
        "company": company,
        "location": location,
        "work_mode": rng.choice(WORK_MODE_SURFACE[mode]) if mode else None,
        "employment_type": rng.choice(EMPLOYMENT_SURFACE[employment]) if employment else None,
        "salary": salary_s,
        "experience": exp_s,
        "skills": skill_s,
    }
    return Spec(label=label, style=style, surfaces=surfaces)


def writer_prompt(spec: Spec) -> str:
    s = spec.surfaces
    must = [f'- job title, written exactly: "{s["title"]}"']
    must.append(f'- company name, written exactly: "{s["company"]}"' if s["company"]
                else "- do NOT name the hiring company (say 'our client' or 'a fast-growing startup')")
    must.append(f'- location, written exactly: "{s["location"]}"' if s["location"]
                else "- do NOT mention any city or country")
    must.append(f'- work arrangement, written exactly: "{s["work_mode"]}"' if s["work_mode"]
                else "- do NOT say whether it is remote, hybrid or in-office")
    must.append(f'- employment type, written exactly: "{s["employment_type"]}"' if s["employment_type"]
                else "- do NOT say whether it is full-time, part-time, contract or an internship")
    must.append(f'- pay, written exactly: "{s["salary"]}"' if s["salary"]
                else "- do NOT mention salary, pay, rates or any money amounts")
    must.append(f'- experience, written exactly: "{s["experience"]}"' if s["experience"]
                else "- do NOT mention a number of years of experience")
    must.append("- these technologies, each written exactly: " + ", ".join(f'"{x}"' for x in s["skills"]))
    return (
        f"Write {STYLES[spec.style]} for a job opening. 50-130 words.\n"
        "It must contain all of the following facts, in any order, woven naturally into the text:\n"
        + "\n".join(must)
        + "\nDo not mention any other programming language, framework, database, cloud or tool. "
        "Do not add any other salary figures or numbers of years. "
        "Output only the post text, no preamble, no quotes around it."
    )
