import csv
import os
import glob
from collections import Counter, defaultdict
from datetime import datetime

# 1. Parse runs_history.csv
with open('bots/runs_history.csv', 'r', encoding='utf-8') as f:
    runs = list(csv.DictReader(f))

print(f"Total runs: {len(runs)}")

# The previous report at 202k steps analyzed up to row 10900.
# Phase 1: 0 - 5639
# Phase 2: 5639 - 8458 (2819 runs)
# Phase 3: 8458 - 10900 (2442 runs)
# Phase 4 (current session): 10900 - 12336 (1436 runs)

p1_runs = runs[:5639]
p2_runs = runs[5639:8458]
p3_runs = runs[8458:10900]
p4_runs = runs[10900:]

print(f"Phase 1 runs: {len(p1_runs)}")
print(f"Phase 2 runs: {len(p2_runs)}")
print(f"Phase 3 runs: {len(p3_runs)}")
print(f"Phase 4 runs: {len(p4_runs)}")

def analyze_phase(name, phase_runs):
    total = len(phase_runs)
    if total == 0:
        return {}
    wins = sum(1 for r in phase_runs if r['win'].strip().lower() == 'true')
    antes = [int(r['max_ante']) for r in phase_runs if r['max_ante'].isdigit()]
    avg_ante = sum(antes) / len(antes) if antes else 0
    max_ante = max(antes) if antes else 0
    moneys = [float(r['max_money']) for r in phase_runs if r['max_money'].replace('.', '', 1).isdigit()]
    max_money = max(moneys) if moneys else 0
    avg_money = sum(moneys) / len(moneys) if moneys else 0
    
    # Boss deaths
    fatal_bosses = Counter(r['fatal_boss'] for r in phase_runs if r['win'].strip().lower() != 'true')
    
    # Most played hands
    played_hands = Counter(r['most_played_hand'] for r in phase_runs if r['most_played_hand'])
    
    # Hand breakdowns
    all_hands = Counter()
    for r in phase_runs:
        hb = r.get('hands_breakdown', '')
        if hb:
            for item in hb.split('|'):
                if ':' in item:
                    hname, hcnt = item.split(':', 1)
                    if hcnt.isdigit():
                        all_hands[hname] += int(hcnt)
                        
    # Jokers bought
    jokers_bought = Counter()
    for r in phase_runs:
        jb = r.get('jokers_bought', '')
        if jb:
            for j in jb.split('|'):
                if j:
                    jokers_bought[j] += 1
                    
    # Ante distribution
    ante_dist = Counter(antes)
    
    # Rolling winrates (window 20 and 50)
    win_list = [1 if r['win'].strip().lower() == 'true' else 0 for r in phase_runs]
    roll20 = [sum(win_list[max(0, i-20):i]) / min(20, i) for i in range(1, len(win_list)+1)] if win_list else [0]
    roll50 = [sum(win_list[max(0, i-50):i]) / min(50, i) for i in range(1, len(win_list)+1)] if win_list else [0]
    max_roll20 = max(roll20) if roll20 else 0
    max_roll50 = max(roll50) if roll50 else 0
    last_quarter = win_list[-len(win_list)//4:] if len(win_list) >= 4 else win_list
    cruising_winrate = sum(last_quarter) / len(last_quarter) if last_quarter else 0

    return {
        'name': name,
        'total': total,
        'wins': wins,
        'winrate': wins / total,
        'cruising_winrate': cruising_winrate,
        'max_roll20': max_roll20,
        'max_roll50': max_roll50,
        'avg_ante': avg_ante,
        'max_ante': max_ante,
        'max_money': max_money,
        'avg_money': avg_money,
        'fatal_bosses': fatal_bosses,
        'all_hands': all_hands,
        'jokers_bought': jokers_bought,
        'ante_dist': ante_dist,
    }

p3_stats = analyze_phase("Phase 3 (141k-202k)", p3_runs)
p4_stats = analyze_phase("Phase 4 (202k-302k)", p4_runs)

print("\n--- PHASE 4 (202k -> 302k) SUMMARY ---")
print(f"Total runs: {p4_stats['total']}")
print(f"Wins: {p4_stats['wins']} ({p4_stats['winrate']*100:.2f}%)")
print(f"Cruising Winrate (last 25%): {p4_stats['cruising_winrate']*100:.2f}%")
print(f"Max rolling 20 winrate: {p4_stats['max_roll20']*100:.2f}%")
print(f"Max rolling 50 winrate: {p4_stats['max_roll50']*100:.2f}%")
print(f"Average max ante: {p4_stats['avg_ante']:.2f}")
print(f"Max money: ${p4_stats['max_money']}")
print(f"Average money: ${p4_stats['avg_money']:.1f}")

print("\n--- ANTE DISTRIBUTION (PHASE 4) ---")
total_p4 = p4_stats['total']
for a in sorted(p4_stats['ante_dist'].keys()):
    cnt = p4_stats['ante_dist'][a]
    print(f"Ante {a}: {cnt} runs ({cnt/total_p4*100:.2f}%)")

print("\n--- FATAL BOSSES COMPARISON (P3 vs P4) ---")
all_bosses = set(list(p3_stats['fatal_bosses'].keys()) + list(p4_stats['fatal_bosses'].keys()))
p3_losses = p3_stats['total'] - p3_stats['wins']
p4_losses = p4_stats['total'] - p4_stats['wins']
print(f"Total losses: P3 = {p3_losses}, P4 = {p4_losses}")
sorted_bosses = sorted(all_bosses, key=lambda b: p4_stats['fatal_bosses'].get(b, 0), reverse=True)
for b in sorted_bosses[:15]:
    c3 = p3_stats['fatal_bosses'].get(b, 0)
    c4 = p4_stats['fatal_bosses'].get(b, 0)
    pct3 = (c3 / p3_losses * 100) if p3_losses else 0
    pct4 = (c4 / p4_losses * 100) if p4_losses else 0
    print(f"{b:25s}: P3={c3:3d} ({pct3:5.1f}%) | P4={c4:3d} ({pct4:5.1f}%)")

print("\n--- TOP JOKERS BOUGHT (PHASE 4) ---")
for j, cnt in p4_stats['jokers_bought'].most_common(20):
    c3 = p3_stats['jokers_bought'].get(j, 0)
    print(f"{j:25s}: P4={cnt:3d} | P3={c3:3d}")

print("\n--- POKER HANDS DISTRIBUTION (PHASE 4) ---")
total_hands_p3 = sum(p3_stats['all_hands'].values())
total_hands_p4 = sum(p4_stats['all_hands'].values())
for h, cnt in p4_stats['all_hands'].most_common(12):
    c3 = p3_stats['all_hands'].get(h, 0)
    pct3 = (c3 / total_hands_p3 * 100) if total_hands_p3 else 0
    pct4 = (cnt / total_hands_p4 * 100) if total_hands_p4 else 0
    print(f"{h:20s}: P4={cnt:5d} ({pct4:5.2f}%) | P3={c3:5d} ({pct3:5.2f}%)")

