from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

log_dir = Path('bots/rl/logs/MaskablePPO_45')
event_files = sorted(log_dir.glob('events.out.tfevents.*'))
print('Found event files in MaskablePPO_45:', len(event_files))

for ef in event_files:
    ea = event_accumulator.EventAccumulator(str(ef))
    ea.Reload()
    tags = ea.Tags()
    scalars = tags.get('scalars', [])
    print(f"\n--- File: {ef.name} ---")
    print(f"Scalars present: {scalars}")
    for tag in ['train/explained_variance', 'rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/approx_kl', 'train/entropy_loss', 'train/policy_gradient_loss', 'train/value_loss', 'rollout/win_rate']:
        if tag in scalars:
            vals = ea.Scalars(tag)
            first_v = vals[0].value
            last_v = vals[-1].value
            min_v = min(v.value for v in vals)
            max_v = max(v.value for v in vals)
            print(f"  {tag:30s}: first={first_v:.4f}, last={last_v:.4f}, min={min_v:.4f}, max={max_v:.4f} (count={len(vals)})")

