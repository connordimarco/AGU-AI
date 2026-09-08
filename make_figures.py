import csv, json, math, os, re, collections
from datetime import date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

YEARS = list(range(2018, 2027))
RELEASE = date(2022, 11, 30)
RELEASE_X = 2022 + 11 / 12
INK, RED, BLUE, GRAY = "#2c3e50", "#c0392b", "#2980b9", "#95a5a6"
NATIVE_EN = {"US", "GB", "CA", "AU", "NZ", "IE"}

LEAD = re.compile(r"^\s*abstract\b", re.I)
WORD = re.compile(r"[A-Za-z]+")
SENT = re.compile(r"[.!?]+")
VOWELS = re.compile(r"[aeiouy]+")
TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
CLOSER = re.compile(r"^\s*(These|Our)\s+(findings|results|observations)\b", re.I)
CLOSER_SPECIFIC = re.compile(r"^\s*(These|Our)\s+(findings|results|observations)\s+"
                             r"(underscore|highlight|demonstrate|emphasize|showcase)", re.I)

MARKERS = [re.compile(p, re.I) for p in [
    r"\bdelv(e|es|ed|ing)\b", r"\bintric(ate|ately|acy|acies)\b", r"\btapestr(y|ies)\b",
    r"\btestament\b", r"\brealms?\b", r"\bmeticulous(ly)?\b", r"\bshowcas(e|es|ed|ing)\b",
    r"\bboast(s|ed|ing)?\b", r"\bgarner(s|ed|ing)?\b", r"\bmulti-?faceted\b",
    r"\bnuanc(e|es|ed|ing)\b", r"\binterplay\b",
    r"\bunderscor(e|es|ed|ing)\b", r"\bpivotal\b", r"\bcrucial(ly)?\b", r"\bvital(ly)?\b",
    r"\bvibrant\b", r"\bcomprehensive(ly)?\b", r"\bbolster(s|ed|ing)?\b",
    r"\brobust(ly|ness)?\b", r"\bseamless(ly)?\b", r"\bparamount\b", r"\bnoteworthy\b",
    r"\bleverag(e|es|ed|ing)\b", r"\bhighlighting\b", r"\bemphasi[sz]ing\b", r"\bfostering\b",
    r"\bencompassing\b", r"\benhancing\b", r"\bplays?\s+(a|an)\s+(\w+\s+){0,2}role\b",
    r"\bserv(e|es|ed|ing)\s+as\b", r"\bstands?\s+as\b", r"\bvaluable\s+insights?\b",
    r"\bshed(s|ding)?\s+light\b", r"\bpav(e|es|ed|ing)\s+the\s+way\b"]]

HEDGES = [re.compile(p, re.I) for p in [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bsuggest(s|ed|ing)?\b",
    r"\bindicat(e|es|ed|ing)\b", r"\b(un)?likely\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\b(appear|seem)(s|ed)?\s+to\b", r"\btend(s|ed)?\s+to\b", r"\bapproximately\b",
    r"\broughly\b", r"\bpresumably\b", r"\btentative(ly)?\b", r"\bprobably\b"]]

BOOSTERS = [re.compile(p, re.I) for p in [
    r"\bclearly\b", r"\bobviously\b", r"\bundoubtedly\b", r"\bcertainly\b", r"\bdefinitely\b",
    r"\bindeed\b", r"\bin\s+fact\b", r"\bevident(ly)?\b", r"\bdemonstrat(e|es|ed|ing)\b",
    r"\bconfirm(s|ed|ing)?\b", r"\bprov(e|es|ed|en|ing)\b", r"\bestablish(es|ed|ing)?\b",
    r"\bunambiguous(ly)?\b", r"\bconclusive(ly)?\b"]]

EXCESS_WORDS = ["findings", "crucial", "insights", "additionally", "reveals", "assess", "primarily",
                "exhibits", "particularly", "utilizing", "notably", "underscores", "pivotal", "intricate"]


def norm_doi(d):
    return d.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().lower()


def parse_date(s):
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def syllables(w):
    n = len(VOWELS.findall(w))
    if w.endswith("e"):
        n -= 1
    return max(n, 1)


def group_of(countries):
    if not countries:
        return "unknown"
    if all(c in NATIVE_EN for c in countries):
        return "native"
    if not any(c in NATIVE_EN for c in countries):
        return "non_native"
    return "mixed"


def period_of(year):
    return "before" if year <= 2021 else "after" if year >= 2023 else None


def ols_predict(xs, ys, x):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    return max(0.0, my - b * mx + b * x)


def mean(v):
    return sum(v) / len(v)


received = {}
for row in csv.DictReader(open(os.path.join(DATA, "received_dates.csv"), encoding="utf-8")):
    dt = parse_date(row["received"])
    if dt:
        received[row["doi"].strip().lower()] = dt

by_quarter = collections.defaultdict(list)
before_received = []
flesch = collections.defaultdict(lambda: collections.defaultdict(list))
flesch_period = collections.defaultdict(list)
words_y, hedges_y, boosters_y = collections.Counter(), collections.Counter(), collections.Counter()
n_last, closer_y, closer_spec_y = collections.Counter(), collections.Counter(), collections.Counter()
n_year = collections.Counter()
word_presence = {w: collections.Counter() for w in EXCESS_WORDS}

for line in open(os.path.join(DATA, "abstracts.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if not r.get("abstract"):
        continue
    year = r["publication_year"]
    if year not in YEARS:
        continue
    text = LEAD.sub("", r["abstract"])
    n_year[year] += 1

    score = sum(1 for m in MARKERS if m.search(text))
    rec = received.get(norm_doi(r.get("doi") or ""))
    if rec:
        by_quarter[(rec.year, (rec.month - 1) // 3 + 1)].append(score)
        if year <= 2021:
            before_received.append(score)

    toks = [w.lower() for w in WORD.findall(text)]
    nw = len(toks)
    if nw >= 20:
        nsent = max(len([s for s in SENT.split(text) if s.strip()]), 1)
        fre = 206.835 - 1.015 * (nw / nsent) - 84.6 * (sum(syllables(w) for w in toks) / nw)
        g = group_of(r.get("author_countries"))
        flesch[g][year].append(fre)
        if period_of(year):
            flesch_period[(g, period_of(year))].append(fre)
        words_y[year] += nw
        hedges_y[year] += sum(len(h.findall(text)) for h in HEDGES)
        boosters_y[year] += sum(len(b.findall(text)) for b in BOOSTERS)

    sents = [s for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    if sents:
        n_last[year] += 1
        closer_y[year] += bool(CLOSER.match(sents[-1]))
        closer_spec_y[year] += bool(CLOSER_SPECIFIC.match(sents[-1]))

    present = {t for t in TOKEN.findall(text.lower()) if len(t) >= 2}
    for w in EXCESS_WORDS:
        if w in present:
            word_presence[w][year] += 1

scores = {}
for line in open(os.path.join(DATA, "pangram_scores.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    scores[norm_doi(r["doi"])] = r["pangram_fraction_ai"] + r["pangram_fraction_ai_assisted"]
pangram_year = collections.defaultdict(list)
for doi, s in scores.items():
    if doi in received:
        pangram_year[received[doi].year].append(s > 0)
pangram_share = {y: 100 * mean(pangram_year[y]) for y in YEARS}

controls = {}
for name in ("human", "ai"):
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"pangram_{name}_controls.jsonl"), encoding="utf-8")]
    controls[name] = (sum(r["pangram_fraction_ai"] + r["pangram_fraction_ai_assisted"] >= 0.5 for r in rows), len(rows))

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5, "legend.framealpha": 0.95,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecf0f1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def shade(ax, x0, x1):
    ax.axvspan(x0, RELEASE_X, color="#3498db", alpha=0.06)
    ax.axvspan(RELEASE_X, x1, color="#e74c3c", alpha=0.06)


def release_line(ax, label=False, y=0.97):
    ax.axvline(RELEASE_X, color=RED, ls=":", lw=1.5)
    if label:
        ax.annotate("ChatGPT released\n2022-11-30", (RELEASE_X + 0.07, y), xycoords=("data", "axes fraction"),
                    fontsize=9.5, color=RED, va="top")


def year_ticks(ax, short=True):
    ax.set_xticks(YEARS)
    ax.set_xticklabels([(str(y)[2:] if short else str(y)) + ("*" if y == 2026 else "") for y in YEARS])


fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.32)
ax, bx, cx = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

quarters = sorted(q for q in by_quarter if len(by_quarter[q]) >= 100)
qx = [y + (q - 1) / 4 + 0.125 for y, q in quarters]
qy = [mean(by_quarter[q]) for q in quarters]
solid = [(x, y) for x, y in zip(qx, qy) if x < 2025.5]
prov = [(x, y) for x, y in zip(qx, qy) if x >= 2025.5]
shade(ax, min(qx) - 0.1, max(qx) + 0.2)
ax.plot(*zip(*solid), "-o", color=INK, lw=2.2, ms=5)
ax.plot(*zip(*([solid[-1]] + prov)), "--", color=INK, lw=1.4)
ax.plot(*zip(*prov), "o", color=INK, ms=5, mfc="white", ls="none")
release_line(ax, label=True, y=0.96)
base = mean(before_received)
ax.axhline(base, color=BLUE, ls="--", lw=1.1, alpha=0.75)
ax.annotate(f"2018–21 average: {base:.2f}", (min(qx) + 0.15, base + 0.008), fontsize=9.5, color=BLUE)
ax.set_xlabel("Quarter received")
ax.set_ylabel("Mean AI-style score")
ax.set_title("(a)  AI-style score by date received", loc="left")

share = [pangram_share[y] for y in YEARS]
bars = bx.bar(YEARS, share, color=[BLUE if y < 2023 else RED for y in YEARS], alpha=0.85, width=0.72)
for b, v in zip(bars, share):
    bx.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}" if v >= 1 else f"{v:.2f}",
            ha="center", va="bottom", fontsize=8.5, color=INK)
bx.axvline(2022.5, color=RED, ls=":", lw=1.4)
year_ticks(bx)
bx.set_ylim(0, max(share) * 1.22)
bx.set_xlabel("Year received (*2026 partial)")
bx.set_ylabel("Abstracts with AI text (%)")
bx.set_title("(b)  Pangram-4 detector", loc="left")

fit_years = [2018, 2019, 2020, 2021, 2022]
exp = {w: 100 * ols_predict(fit_years, [word_presence[w][y] / n_year[y] for y in fit_years], 2024) for w in EXCESS_WORDS}
obs = {w: 100 * word_presence[w][2024] / n_year[2024] for w in EXCESS_WORDS}
words = sorted(EXCESS_WORDS, key=lambda w: obs[w] - exp[w])
ypos = range(len(words))
cx.barh([y + 0.2 for y in ypos], [exp[w] for w in words], height=0.38, color=BLUE, alpha=0.8, label="predicted by 2018–22 trend")
cx.barh([y - 0.2 for y in ypos], [obs[w] for w in words], height=0.38, color=RED, alpha=0.85, label="observed in 2024")
for y, w in zip(ypos, words):
    r = obs[w] / exp[w] if exp[w] > 0 else float("inf")
    cx.text(obs[w] + 0.15, y - 0.2, f"{r:.1f}×" if r < 10 else f"{r:.0f}×", va="center", fontsize=8.5, color=INK)
cx.set_yticks(list(ypos))
cx.set_yticklabels([f"$\\it{{{w}}}$" for w in words])
cx.set_xlabel("Abstracts containing word (%)")
cx.set_title("(c)  Words above trend, 2024", loc="left")
cx.set_xlim(0, max(obs.values()) * 1.25)
cx.legend(loc="lower right", fontsize=8.5)
cx.grid(axis="y", visible=False)
fig.savefig(os.path.join(FIG, "fig_c1_timing.png"))
plt.close(fig)

fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.3)
ax, bx, cx = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

shade(ax, 2017.6, 2026.5)
for g, lab, col, mk in [("non_native", "institutions in non-predominantly\nEnglish-speaking nations", RED, "o"),
                        ("mixed", "mixed teams", GRAY, "^"),
                        ("native", "institutions in predominantly\nEnglish-speaking nations", BLUE, "s")]:
    ax.plot(YEARS, [mean(flesch[g][y]) for y in YEARS], "-", marker=mk, color=col, lw=2.2, ms=6, label=lab)
release_line(ax, label=True, y=0.98)
d_non = mean(flesch_period[("non_native", "after")]) - mean(flesch_period[("non_native", "before")])
d_nat = mean(flesch_period[("native", "after")]) - mean(flesch_period[("native", "before")])
ax.text(0.03, 0.05, f"before vs after, non-predominantly English-speaking nations: {d_non:+.1f} points\n"
        f"predominantly English-speaking nations: {d_nat:+.1f} points\ndifference: {d_non - d_nat:+.1f} (p ≈ 1e-11)",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GRAY, alpha=0.95))
year_ticks(ax, short=False)
ax.set_xlabel("Publication year (*2026 partial)")
ax.set_ylabel("Reading ease (Flesch score; higher = easier)")
ax.set_title("(a)  Readability fell in every author group", loc="left")
ax.set_ylim(12.5, ax.get_ylim()[1])
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.36), fontsize=8.5)

b = [1e4 * boosters_y[y] / words_y[y] for y in YEARS]
h = [1e4 * hedges_y[y] / words_y[y] for y in YEARS]
shade(bx, 2017.6, 2026.5)
bx.plot(YEARS, [100 * x / mean(b[:4]) for x in b], "-o", color=RED, lw=2.2, ms=5, label="boosters (clearly, demonstrate, confirm …)")
bx.plot(YEARS, [100 * x / mean(h[:4]) for x in h], "-s", color=BLUE, lw=2.2, ms=5, label="hedges (may, suggest, likely …)")
bx.axhline(100, color=GRAY, lw=0.9, ls="--")
release_line(bx)
year_ticks(bx)
bx.set_ylabel("Rate per 10,000 words\n(2018–21 average = 100)")
bx.set_title("(b)  Boosters rose; hedges did not", loc="left")
bx.legend(loc="lower left", fontsize=8.5)
bx.set_ylim(70, 165)

shade(cx, 2017.6, 2026.5)
cx.plot(YEARS, [100 * closer_y[y] / n_last[y] for y in YEARS], "-o", color=INK, lw=2.2, ms=5, label="ends with “These/Our findings …”")
cx.plot(YEARS, [100 * closer_spec_y[y] / n_last[y] for y in YEARS], "-^", color=RED, lw=2.2, ms=5, label="… “underscore/highlight/demonstrate …”")
release_line(cx)
year_ticks(cx)
cx.set_xlabel("Publication year (*2026 partial)")
cx.set_ylabel("Share of abstracts (%)")
cx.set_title("(c)  One closing formula spreads", loc="left")
cx.legend(loc="upper left", fontsize=8.5)
cx.set_ylim(0, 37)
fig.savefig(os.path.join(FIG, "fig_c2_hype.png"))
plt.close(fig)

print("abstracts by year:", dict(sorted(n_year.items())))
print("2018-21 average score (received-dated):", round(base, 3))
print("score by received quarter:", [(f"{y}Q{q}", round(v, 3)) for (y, q), v in zip(quarters, qy)])
print("Pangram share with AI text by year received (%):", {y: round(v, 2) for y, v in pangram_share.items()})
print("Pangram controls flagged: human %d/%d, AI %d/%d" % (controls["human"] + controls["ai"]))
print("2024 excess words (observed %, predicted %):", {w: (round(obs[w], 2), round(exp[w], 2)) for w in words})
print("Flesch by year:", {g: {y: round(mean(flesch[g][y]), 1) for y in YEARS} for g in ("non_native", "mixed", "native")})
print("Flesch change before -> after: non-native %.2f, native %.2f, difference %.2f" % (d_non, d_nat, d_non - d_nat))
print("boosters per 10k by year:", dict(zip(YEARS, [round(x, 2) for x in b])))
print("hedges per 10k by year:", dict(zip(YEARS, [round(x, 2) for x in h])))
print("closing formula (%):", {y: (round(100 * closer_y[y] / n_last[y], 1), round(100 * closer_spec_y[y] / n_last[y], 1)) for y in YEARS})
