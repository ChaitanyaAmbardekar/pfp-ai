"""Turn a best-to-worst ranking into pairwise labels (winner beats every later photo)."""
import csv
import itertools
from pathlib import Path

RANKING = ["s2", "s4", "s3", "s8", "s6", "s7", "s5"]   # best -> worst. Edit if s5 is wrong.

out = Path("dataset/labels")
out.mkdir(parents=True, exist_ok=True)
with open(out / "pairwise.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["image_A", "image_B", "preferred", "source"])
    for a, b in itertools.combinations(RANKING, 2):   # a is ranked above b
        w.writerow([a, b, a, "manual_ranking_1"])
print(f"wrote {len(RANKING) * (len(RANKING) - 1) // 2} pairs to {out / 'pairwise.csv'}")
