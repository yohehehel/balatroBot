import csv
from collections import Counter

with open('bots/runs_history.csv', 'r', encoding='utf-8') as f:
    runs = list(csv.DictReader(f))

# Phase 4 starts at row 8492 (2026-10-31 12:40:07)
p4_runs = runs[8492:]
print(f"Total runs in Phase 4 (100k steps): {len(p4_runs)}")

# Divide Phase 4 into 4 quartiles
n = len(p4_runs)
q_size = n // 4
quartiles = [
    ("Quartile 1 (Steps 202k-227k)", p4_runs[0:q_size]),
    ("Quartile 2 (Steps 227k-252k)", p4_runs[q_size:2*q_size]),
    ("Quartile 3 (Steps 252k-277k)", p4_runs[2*q_size:3*q_size]),
    ("Quartile 4 (Steps 277k-303k)", p4_runs[3*q_size:]),
]

for name, q_runs in quartiles:
    wins = sum(1 for r in q_runs if r['win'].strip().lower() == 'true')
    total = len(q_runs)
    antes = [int(r['max_ante']) for r in q_runs if r['max_ante'].isdigit()]
    avg_ante = sum(antes) / len(antes) if antes else 0
    ante9 = sum(1 for a in antes if a >= 9)
    print(f"\n--- {name} ({total} runs) ---")
    print(f"Wins: {wins}/{total} ({wins/total*100:.2f}%)")
    print(f"Average Ante: {avg_ante:.2f}")
    print(f"Ante 9 (Endless/Win): {ante9}/{total} ({ante9/total*100:.2f}%)")
    ante_dist = Counter(antes)
    print(f"Ante 1-2 mortality: {ante_dist.get(1,0)+ante_dist.get(2,0)} ({ (ante_dist.get(1,0)+ante_dist.get(2,0))/total*100:.2f}%)")
    print(f"Ante 5 deaths: {ante_dist.get(5,0)} ({ante_dist.get(5,0)/total*100:.2f}%)")
    print(f"Ante 6 deaths: {ante_dist.get(6,0)} ({ante_dist.get(6,0)/total*100:.2f}%)")

