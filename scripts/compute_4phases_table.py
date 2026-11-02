import csv
from pathlib import Path
from collections import Counter

runs_file = Path("bots/runs_history.csv")
with open(runs_file, "r", encoding="utf-8") as f:
    runs = list(csv.DictReader(f))

rounds_file = Path("bots/rounds_history.csv")
with open(rounds_file, "r", encoding="utf-8") as f:
    rounds = list(csv.DictReader(f))

p1_runs = [r for r in runs if r.get("timestamp", "") < "2026-10-29 21:00"]
p2_runs = [r for r in runs if "2026-10-29 21:00" <= r.get("timestamp", "") < "2026-10-30 12:30"]
p3_runs = [r for r in runs if "2026-10-30 12:30" <= r.get("timestamp", "") < "2026-10-31 12:35"]
p4_runs = [r for r in runs if r.get("timestamp", "") >= "2026-10-31 12:35"]

p1_rounds = [r for r in rounds if r.get("timestamp", "") < "2026-10-29 21:00"]
p2_rounds = [r for r in rounds if "2026-10-29 21:00" <= r.get("timestamp", "") < "2026-10-30 12:30"]
p3_rounds = [r for r in rounds if "2026-10-30 12:30" <= r.get("timestamp", "") < "2026-10-31 12:35"]
p4_rounds = [r for r in rounds if r.get("timestamp", "") >= "2026-10-31 12:35"]

def get_stats(run_list, round_list):
    antes = [int(r["max_ante"]) for r in run_list if r.get("max_ante") and r["max_ante"].isdigit()]
    wins = [1 if r.get("win", "").lower() in ("true", "1") else 0 for r in run_list]
    scores = [int(r.get("total_score", 0) or 0) for r in round_list if r.get("outcome") == "WON"]
    moneys = [int(float(r.get("max_money", 0) or r.get("final_money", 0) or 0)) for r in run_list]
    rewards = [float(r.get("total_reward", 0.0) or 0.0) for r in run_list]
    
    # Calculate rolling win rates
    roll20 = [sum(wins[max(0, i-20):i]) / min(20, i) for i in range(1, len(wins)+1)] if wins else [0]
    roll50 = [sum(wins[max(0, i-50):i]) / min(50, i) for i in range(1, len(wins)+1)] if wins else [0]
    
    # Cruising speed (last quarter)
    last_quarter = wins[-len(wins)//4:] if len(wins) >= 4 else wins
    cruising_wr = sum(last_quarter) / len(last_quarter) * 100 if last_quarter else 0

    return {
        "runs": len(run_list),
        "rounds": len(round_list),
        "wins": sum(wins),
        "winrate": sum(wins) / len(run_list) * 100 if run_list else 0,
        "cruising_winrate": cruising_wr,
        "max_roll20": max(roll20) * 100 if roll20 else 0,
        "max_roll50": max(roll50) * 100 if roll50 else 0,
        "avg_ante": sum(antes) / len(antes) if antes else 0,
        "max_score": max(scores) if scores else 0,
        "avg_won_score": sum(scores) / len(scores) if scores else 0,
        "max_money": max(moneys) if moneys else 0,
        "avg_max_money": sum(moneys) / len(moneys) if moneys else 0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
    }

s1 = get_stats(p1_runs, p1_rounds)
s2 = get_stats(p2_runs, p2_rounds)
s3 = get_stats(p3_runs, p3_rounds)
s4 = get_stats(p4_runs, p4_rounds)
stot = get_stats(runs, rounds)

print("=== CONSOLIDATED 4-PHASE SUMMARY TABLE ===")
print(f"Total runs across all time: {stot['runs']}")
print(f"Total rounds across all time: {stot['rounds']}")
print(f"Total wins across all time: {stot['wins']}")
print(f"Global Winrate across all time: {stot['winrate']:.2f}%")
print()
print(f"{'Métrique Clé':<35} | {'Phase 1 (0-80k)':<15} | {'Phase 2 (80-141k)':<17} | {'Phase 3 (141-202k)':<18} | {'Phase 4 (202-303k)':<18}")
print("-" * 115)
print(f"{'Volume de Timesteps':<35} | {'106 392 pas':<15} | {'141 246 pas':<17} | {'202 626 pas':<18} | {'302 978 pas (+100.3k)':<18}")
print(f"{'Parties Disputées':<35} | {s1['runs']:<15} | {s2['runs']:<17} | {s3['runs']:<18} | {s4['runs']:<18}")
print(f"{'Rounds / Blinds Joués':<35} | {s1['rounds']:<15} | {s2['rounds']:<17} | {s3['rounds']:<18} | {s4['rounds']:<18}")
print(f"{'Victoires Totales (Ante 8+)':<35} | {s1['wins']:<15} | {s2['wins']:<17} | {s3['wins']:<18} | {s4['wins']:<18}")
print(f"{'Taux de Victoire Réel':<35} | {s1['winrate']:<14.2f}% | {s2['winrate']:<16.2f}% | {s3['winrate']:<17.2f}% | {s4['winrate']:<17.2f}%")
print(f"{'Vitesse de Croisière (Fin)':<35} | {'~5.0%':<15} | {'~45.0%':<17} | {s3['cruising_winrate']:<17.2f}% | {s4['cruising_winrate']:<17.2f}%")
print(f"{'Pic Win Rate Glissant (20 runs)':<35} | {'25.0%':<15} | {'65.0%':<17} | {s3['max_roll20']:<17.1f}% | {s4['max_roll20']:<17.1f}%")
print(f"{'Pic Win Rate Glissant (50 runs)':<35} | {'N/A':<15} | {'N/A':<17} | {s3['max_roll50']:<17.1f}% | {s4['max_roll50']:<17.1f}%")
print(f"{'Moyenne d Ante Maximale':<35} | {s1['avg_ante']:<15.2f} | {s2['avg_ante']:<17.2f} | {s3['avg_ante']:<18.2f} | {s4['avg_ante']:<18.2f}")
print(f"{'Score Record en un Round':<35} | {s1['max_score']:<15,d} | {s2['max_score']:<17,d} | {s3['max_score']:<18,d} | {s4['max_score']:<18,d}")
print(f"{'Plafond d Or Atteint (Max Money)':<35} | ${s1['max_money']:<14} | ${s2['max_money']:<16} | ${s3['max_money']:<17} | ${s4['max_money']:<17}")
print(f"{'Moyenne d Or Possédé':<35} | ${s1['avg_max_money']:<14.1f} | ${s2['avg_max_money']:<16.1f} | ${s3['avg_max_money']:<17.1f} | ${s4['avg_max_money']:<17.1f}")
print(f"{'Récompense Moyenne / Run':<35} | {s1['avg_reward']:<15.2f} | {s2['avg_reward']:<17.2f} | {s3['avg_reward']:<18.2f} | {s4['avg_reward']:<18.2f}")

# Phase 4 detailed cruising speed (final 500 runs)
p4_last500 = p4_runs[-500:]
p4_last500_wins = sum(1 if r.get('win', '').lower() in ('true', '1') else 0 for r in p4_last500)
p4_last500_antes = [int(r['max_ante']) for r in p4_last500 if r.get('max_ante') and r['max_ante'].isdigit()]
print(f"\nPhase 4 - Dernières 500 parties :")
print(f"Wins: {p4_last500_wins} / 500 ({p4_last500_wins / 500 * 100:.2f}%)")
print(f"Avg Max Ante: {sum(p4_last500_antes) / len(p4_last500_antes):.2f}")

