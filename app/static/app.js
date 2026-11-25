const $ = (s, el = document) => el.querySelector(s);
const FIELDS = ["title", "company", "location", "work_mode", "employment_type", "seniority", "salary", "experience_years_min", "skills"];
let samples = [], current = null;

/* ---------- hero animation: fragments fall through the mesh and land as fields ---------- */
const FRAGS = ["🚀 we're HIRING!!", "₹18-24 LPA", "5+ yrs", "hybrid, 2 days WFH", "K8s", "our client", "#hiring",
  "Sr. Data Engineer", "€55.000 – €65.000", "DM me", "React.js", "6-month contract", "Berlin, Germany", "Postgres", "/hr", "fully remote"];
const LANDED = [["title", "\"Senior Data Engineer\""], ["work_mode", "\"hybrid\""], ["salary.min", "1800000"],
  ["currency", "\"INR\""], ["skills", "[\"kubernetes\", \"postgresql\"]"], ["experience_years_min", "5"]];

function heroLoop() {
  const fall = $("#fall"), sorted = $("#sorted");
  let i = 0, j = 0;
  const drop = () => {
    const f = document.createElement("span");
    f.className = "frag";
    f.textContent = FRAGS[i++ % FRAGS.length];
    f.style.left = (4 + Math.random() * 46) + "%";
    f.style.animationDuration = (2.2 + Math.random() * 1.6) + "s";
    fall.appendChild(f);
    f.addEventListener("animationend", () => f.remove());
  };
  const land = () => {
    const [k, v] = LANDED[j++ % LANDED.length];
    const row = document.createElement("div");
    row.className = "kv";
    row.innerHTML = `<b>${k}</b><span>${v}</span>`;
    sorted.appendChild(row);
    while (sorted.children.length > 5) sorted.firstChild.remove();
  };
  for (let n = 0; n < 4; n++) setTimeout(drop, n * 250);
  setInterval(drop, 520);
  setTimeout(() => { land(); setInterval(land, 1300); }, 900);
}

/* ---------- count-up stats + leaderboard ---------- */
function countUp(el, target, suffix = "%") {
  const t0 = performance.now(), dur = 1200;
  const tick = (t) => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(target * e) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const pct = (x) => x == null ? "–" : (x * 100).toFixed(1) + "%";

async function loadResults() {
  const rows = await fetch("/api/results").then(r => r.json()).catch(() => []);
  if (!rows.length) { $("#board-sub").textContent = "Run the eval harness to fill this table."; return; }
  const ft = rows.find(r => r.highlight) || rows[rows.length - 1];
  const base = rows.find(r => r.key === "base-zero") || rows[0];
  setTimeout(() => {
    countUp($('[data-k="ft_exact"]'), ft.exact_match * 100);
    countUp($('[data-k="base_exact"]'), base.exact_match * 100);
    countUp($('[data-k="ft_valid"]'), ft.schema_valid * 100);
  }, 600);
  $("#board-sub").textContent = `${ft.n} held-out posts the model never saw. Greedy decoding, same system prompt for every row.`;
  const cols = [["label", "Model"], ["exact_match", "Exact match"], ["fact_f1", "Field F1"], ["schema_valid", "Valid JSON"],
    ["latency_ms_p50", "p50 latency"], ["peak_vram_gb", "VRAM"]];
  const head = "<tr>" + cols.map(c => `<th>${c[1]}</th>`).join("") + "</tr>";
  const body = rows.map(r => `<tr class="${r.highlight ? "hl" : ""}">` + cols.map(([k]) => {
    const v = r[k];
    if (k === "label") return `<td>${v}</td>`;
    if (k === "latency_ms_p50") return `<td>${v == null ? "–" : Math.round(v) + " ms"}</td>`;
    if (k === "peak_vram_gb") return `<td>${v == null ? "–" : v.toFixed(1) + " GB"}</td>`;
    return `<td class="bar-cell">${pct(v)}<i style="width:${(v || 0) * 100}%"></i></td>`;
  }).join("") + "</tr>").join("");
  $("#board").innerHTML = head + body;
}

/* ---------- samples + extraction ---------- */
async function loadSamples() {
  samples = await fetch("/api/samples").then(r => r.json()).catch(() => []);
  const chips = $("#chips");
  samples.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = `${s.style.replace("_", " ")} · ${s.label.title}`;
    b.onclick = () => {
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("on"));
      b.classList.add("on");
      current = s;
      $("#text").value = s.text;
    };
    chips.appendChild(b);
    if (i === 0) b.click();
  });
}

function fmt(v) {
  if (v == null) return `<span class="v null">null</span>`;
  const s = typeof v === "object" ? JSON.stringify(v) : String(v);
  return `<span class="v">${s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]))}</span>`;
}

function render(id, side) {
  const el = $(id);
  const badges = [
    `<span class="badge ${side.schema_valid ? "ok" : "bad"}">${side.schema_valid ? "schema ✓" : side.json_valid ? "json, bad schema" : "invalid json"}</span>`,
    `<span class="badge">${side.latency_ms} ms</span>`];
  if (side.fields_correct) {
    const n = Object.values(side.fields_correct).filter(Boolean).length;
    badges.unshift(`<span class="badge ${n === FIELDS.length ? "ok" : "bad"}">${n}/${FIELDS.length} fields</span>`);
  }
  $(".badges", el).innerHTML = badges.join("");
  $(".raw", el).textContent = side.raw;
  const p = side.parsed;
  if (!p) { $(".fields", el).innerHTML = `<p class="error">No JSON object could be parsed from the output.</p>`; return; }
  $(".fields", el).innerHTML = FIELDS.map((k, i) => {
    const ok = side.fields_correct ? side.fields_correct[k] : null;
    const mk = ok == null ? "" : ok ? `<span class="mk y">✓</span>` : `<span class="mk n">✕</span>`;
    return `<div class="row" style="--i:${i}"><span class="k">${k}</span>${fmt(p[k])}${mk}</div>`;
  }).join("");
}

async function extract() {
  const text = $("#text").value.trim();
  if (text.length < 10) return;
  const gold = current && current.text === $("#text").value ? current.label : null;
  const go = $("#go");
  go.classList.add("busy");
  ["#side-base", "#side-ft"].forEach(id => {
    $(".fields", $(id)).innerHTML = Array(6).fill('<div class="skeleton"></div>').join("");
    $(".badges", $(id)).innerHTML = "";
  });
  try {
    const r = await fetch("/api/extract", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, gold }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    render("#side-base", data.base);
    render("#side-ft", data.finetuned);
  } catch (e) {
    ["#side-base", "#side-ft"].forEach(id => $(".fields", $(id)).innerHTML = `<p class="error">${e.message}</p>`);
  } finally {
    go.classList.remove("busy");
  }
}

$("#go").onclick = extract;
$("#text").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) extract(); });
$("#text").addEventListener("input", () => { if (current && current.text !== $("#text").value) document.querySelectorAll(".chip").forEach(c => c.classList.remove("on")); });
heroLoop();
loadResults();
loadSamples();
