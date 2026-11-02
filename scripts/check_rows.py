import csv
with open('bots/runs_history.csv', encoding='utf-8') as f:
    runs = list(csv.DictReader(f))

print(f"Total runs: {len(runs)}")
for i in range(8400, len(runs), 300):
    r = runs[i]
    print(f"Row {i:5d}: {r['timestamp']} | win={r['win']} | ante={r['max_ante']}")

