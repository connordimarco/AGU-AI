import csv, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "word_changes.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

YEARS = list(range(2018, 2027))
LEAD = re.compile(r"^\s*abstract\b", re.I)
TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
MIN_ABSTRACTS = 10
T975 = {2: 4.3027, 3: 3.1824}

n_year = collections.Counter()
presence = collections.defaultdict(collections.Counter)
for line in open(os.path.join(DATA, "abstracts.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if not r.get("abstract") or r["publication_year"] not in YEARS:
        continue
    n_year[r["publication_year"]] += 1
    for w in {t for t in TOKEN.findall(LEAD.sub("", r["abstract"]).lower()) if len(t) >= 2}:
        presence[w][r["publication_year"]] += 1

words = sorted(w for w, c in presence.items() if sum(c.values()) >= MIN_ABSTRACTS)
n = np.array([n_year[y] for y in YEARS], dtype=float)
P = np.array([[presence[w][y] for y in YEARS] for w in words]) / n


def scan(fit_years, target):
    cols = [YEARS.index(y) for y in fit_years]
    x = np.array(fit_years, dtype=float)
    Y = P[:, cols]
    k = len(fit_years)
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()
    slope = ((x - xbar) * (Y - Y.mean(axis=1, keepdims=True))).sum(axis=1) / sxx
    intercept = Y.mean(axis=1) - slope * xbar
    resid = Y - (intercept[:, None] + slope[:, None] * x[None, :])
    s = np.sqrt((resid ** 2).sum(axis=1) / (k - 2))
    exp = np.clip(intercept + slope * target, 0, None)
    obs = P[:, YEARS.index(target)]
    se = s * np.sqrt(1 + 1 / k + (target - xbar) ** 2 / sxx)
    lo = np.clip(exp - T975[k - 2] * se, 0, None)
    hi = exp + T975[k - 2] * se
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(exp > 0, obs / exp, np.inf)
    flagged = ((obs < lo) | (obs > hi)) & (obs - exp >= 0.002) & (ratio >= 1.5)
    return exp, obs, lo, hi, ratio, flagged


exp24, obs24, lo24, hi24, ratio24, flag24 = scan([2018, 2019, 2020, 2021, 2022], 2024)
exp22, obs22, lo22, hi22, ratio22, flag22 = scan([2018, 2019, 2020, 2021], 2022)

order = np.argsort(obs24 - exp24)[::-1]
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["word", "abstracts_total"] + [f"share_{y}_%" for y in YEARS] +
               ["predicted_2024_%", "observed_2024_%", "change_2024_pp", "ratio_2024",
                "interval_low_%", "interval_high_%", "above_trend_2024", "above_trend_2022_placebo"])
    for i in order:
        w.writerow([words[i], int(sum(presence[words[i]].values()))] +
                   [f"{100 * v:.3f}" for v in P[i]] +
                   [f"{100 * exp24[i]:.3f}", f"{100 * obs24[i]:.3f}", f"{100 * (obs24[i] - exp24[i]):+.3f}",
                    "inf" if np.isinf(ratio24[i]) else f"{ratio24[i]:.3f}",
                    f"{100 * lo24[i]:.3f}", f"{100 * hi24[i]:.3f}", bool(flag24[i]), bool(flag22[i])])

print("words tested:", len(words))
print("above trend in 2024:", int(flag24.sum()), " in the 2022 placebo:", int(flag22.sum()))
print("wrote", OUT)
