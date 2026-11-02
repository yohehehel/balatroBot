from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

log_file = Path('bots/rl/logs/MaskablePPO_45/events.out.tfevents.1793446792.DESKTOP-GK4L0FV.12160.0')
ea = event_accumulator.EventAccumulator(str(log_file))
ea.Reload()

for tag in ['game/win_rate_rolling20', 'game/win_rate_rolling50', 'game/avg_ante_rolling20', 'game/max_money', 'rollout/success_rate']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        print(f"\nTag: {tag} (steps={len(vals)})")
        print(f"  First: {vals[0].value:.4f} at step {vals[0].step}")
        print(f"  Last : {vals[-1].value:.4f} at step {vals[-1].step}")
        print(f"  Max  : {max(v.value for v in vals):.4f}")
        print(f"  Min  : {min(v.value for v in vals):.4f}")
        # Print every 5th value
        sampled = [(v.step, round(v.value, 4)) for i, v in enumerate(vals) if i % 5 == 0 or i == len(vals)-1]
        print(f"  Sampled: {sampled}")

