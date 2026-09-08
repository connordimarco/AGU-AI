# Code and data for "Large Language Models Shifted Geophysics Abstracts Toward Emphasis, Not Clarity"

C. DiMarco, T. Pulkkinen, S. Hill. Commentary submitted to *JGR: Space Physics*.

`python3 make_figures.py` reads `data/` and writes the paper's two figures to `figures/`.
It also prints the numbers quoted in the text. Needs Python 3 and matplotlib.

```
data/abstracts.jsonl              21,762 articles from GRL, JGR Space Physics, and Space Weather,
                                  2018 to mid-2026, one JSON record per line (OpenAlex metadata, CC0):
                                  title, abstract, publication date, authors, author countries, topics
data/received_dates.csv           Crossref received and accepted dates by DOI
data/pangram_scores.jsonl         Pangram-4 score per DOI (fraction AI, AI-assisted, human)
data/pangram_human_controls.jsonl Pangram-4 scores for 300 pre-LLM abstracts from these journals
data/pangram_ai_controls.jsonl    Pangram-4 scores for 30 LLM-written test abstracts
```

Definitions used in `make_figures.py`: the AI-style score is the number of distinct
marker patterns present in an abstract (35 regular expressions in the script). Author
groups use the country of each author's institution; the English-speaking set is
US, GB, CA, AU, NZ, IE. Flesch reading ease, hedges, and boosters are computed on
abstracts of at least 20 words.
