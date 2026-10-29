"""CLI entry-point for the Balatro bot (Silo 1 skeleton).

Usage examples
--------------
# Dump the current live game state to a fixture file
uv run python -m bots.cli --dump-state blind_select.json

# Test Silo 1: start a Red Deck / White Stake run and enter the first combat
uv run python -m bots.cli --test-silo1

# With fast mode (no delay between actions)
uv run python -m bots.cli --test-silo1 --fast

# Custom port
uv run python -m bots.cli --test-silo1 --port 12347
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bots.client_wrapper import BalatroClient, ClientConfig, RPCError
from bots.state_parser import (
    GameScreen,
    fetch_gamestate,
    navigate_to_combat,
    parse_gamestate,
)

# Default directory for dumped fixtures
FIXTURES_DIR = Path(__file__).resolve().parent / "tests" / "fixtures"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bots.cli",
        description="Balatro heuristic bot – Silo 1 CLI skeleton",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12346,
        help="Port the Balatro API is listening on (default: 12346)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Delay in seconds between API calls (default: 0.8)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable inter-action delay (equivalent to --delay 0.0)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of complete runs to execute (default: 1)",
    )
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Continue playing past Ante 8 in endless mode until Game Over",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Optional seed for deterministic runs",
    )
    parser.add_argument(
        "--agent",
        choices=["heuristic", "rl"],
        default="heuristic",
        help="Decision agent architecture to use: heuristic (default) or rl",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="bots/rl/checkpoints/best_model.zip",
        help="Path to trained MaskablePPO model checkpoint for --agent rl (default: bots/rl/checkpoints/best_model.zip)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display telemetry and session statistics from runs_history.csv",
    )

    # ── Silo 1 actions ─────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dump-state",
        metavar="FILENAME",
        help="Dump the current live game state to bots/tests/fixtures/FILENAME",
    )
    group.add_argument(
        "--test-silo1",
        action="store_true",
        help="Start a Red Deck / White Stake run and select the first blind",
    )
    group.add_argument(
        "--test-silo2",
        action="store_true",
        help="Test Silo 2: Evaluate hand combinations and play optimal hand",
    )
    group.add_argument(
        "--test-silo3",
        action="store_true",
        help="Test Silo 3: Full combat decision loop (targeted discard and play)",
    )
    group.add_argument(
        "--test-silo4",
        action="store_true",
        help="Test Silo 4: Economy engine, shop management, and joker positioning",
    )
    group.add_argument(
        "--test-silo5",
        action="store_true",
        help="Test Silo 5: Booster packs and consumables management",
    )
    return parser


def _make_client(args: argparse.Namespace) -> BalatroClient:
    delay = 0.0 if args.fast else args.delay
    config = ClientConfig(
        base_url=f"http://localhost:{args.port}",
        delay=delay,
    )
    return BalatroClient(config)


# ── Actions ───────────────────────────────────────────────────────────────

def _action_dump_state(client: BalatroClient, filename: str) -> None:
    """Dump the live game state to a JSON fixture file."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIXTURES_DIR / filename

    raw = client.call("gamestate")
    dest.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ State dumped to {dest}")


def _action_test_silo1(client: BalatroClient) -> None:
    """Run the Silo 1 live test sequence."""
    print("── Silo 1 live test ──────────────────────────────────")

    # 1. Fetch current state
    state = fetch_gamestate(client)
    print(f"  Current screen : {state.state}")

    # 2. Navigate to combat
    state = navigate_to_combat(client)

    print(f"  After navigate : {state.state}")
    print(f"  Deck / Stake   : {state.deck} / {state.stake}")
    print(f"  Ante / Round   : {state.ante_num} / {state.round_num}")
    print(f"  Money          : ${state.money}")

    if state.state == GameScreen.SELECTING_HAND:
        print(f"  Cards in hand  : {state.hand.count}")
        print("✓ Successfully entered combat — Silo 1 validated.")
    else:
        print(f"⚠  Expected SELECTING_HAND but got {state.state}")
        sys.exit(1)


def _action_test_silo2(client: BalatroClient) -> None:
    """Run the Silo 2 live test sequence: evaluate all plays, display Top 3, and play best."""
    from bots.combat_solver import evaluate_all_plays, select_best_play

    print("── Silo 2 live test (Combat Solver) ─────────────────")

    # 1. Ensure we are in combat (SELECTING_HAND)
    state = fetch_gamestate(client)
    if state.state in (GameScreen.MENU, GameScreen.BLIND_SELECT):
        print(f"  Current screen is {state.state}, navigating to combat...")
        state = navigate_to_combat(client)

    if state.state != GameScreen.SELECTING_HAND:
        print(f"✗ Cannot test Silo 2: Game screen is {state.state}, expected SELECTING_HAND.")
        sys.exit(1)

    active_blind = state.active_blind
    target_score = active_blind.score if active_blind else 300
    chips_needed = max(0, target_score - state.round.chips)

    print(f"  Blind          : {active_blind.name if active_blind else 'Unknown'} ({active_blind.type if active_blind else ''})")
    print(f"  Chips needed   : {chips_needed} (Current: {state.round.chips} / Target: {target_score})")
    print(f"  Hands left     : {state.round.hands_left} | Discards left: {state.round.discards_left}")
    print(f"  Cards in hand  : {state.hand.count}")
    if state.jokers.cards:
        joker_names = [j.label or j.key for j in state.jokers.cards]
        print(f"  Jokers         : {', '.join(joker_names)}")

    # 2. Evaluate all combinations
    candidates = evaluate_all_plays(state)
    if not candidates:
        print("✗ No valid play combinations found in hand.")
        sys.exit(1)

    print(f"\n  Total combinations evaluated : {len(candidates)}")
    print("  ── Top 3 Candidate Plays ──────────────────────────")
    for rank_idx, c in enumerate(candidates[:3], 1):
        labels_str = ", ".join(c.card_labels)
        print(
            f"   #{rank_idx} [{c.hand_type}] ({c.final_chips} chips × {c.final_mult:.1f} mult) "
            f"→ Predicted: {c.score:,} pts\n"
            f"       Cards ({len(c.indices)}): {labels_str}"
        )

    # 3. Select best play
    best = select_best_play(state)
    assert best is not None

    kill_shot = best.score >= chips_needed
    strategy_note = "KILL SHOT (optimal card conservation)" if kill_shot else "MAX SCORE"
    print(f"\n  ── Selected Decision ({strategy_note}) ──────────")
    print(f"  Hand Type      : {best.hand_type}")
    print(f"  Cards to play  : {', '.join(best.card_labels)} (indices: {list(best.indices)})")
    print(f"  Predicted Score: {best.score:,} pts ({best.final_chips} chips × {best.final_mult:.1f} mult)")

    # 4. Play the hand via API
    old_chips = state.round.chips
    print("\n  Executing 'play' action...")
    raw_new = client.call("play", {"cards": list(best.indices)})
    new_state = parse_gamestate(raw_new)

    # 5. Compare with actual score
    if new_state.state == GameScreen.ROUND_EVAL:
        actual_score = new_state.round.chips - old_chips if new_state.round.chips > old_chips else target_score - old_chips
    else:
        actual_score = new_state.round.chips - old_chips

    diff = actual_score - best.score
    diff_str = f"+{diff}" if diff > 0 else str(diff)

    print(f"  After play state : {new_state.state}")
    print(f"  Actual Score     : {actual_score:,} pts")
    print(f"  Predicted Score  : {best.score:,} pts")
    print(f"  Difference       : {diff_str} pts")

    if diff == 0:
        print("✓ Precision check passed: Exact score match (0 diff)!")
    elif abs(diff) <= 2:
        print(f"✓ Precision check passed: Close match ({diff_str} pts difference due to rounding)")
    else:
        print(f"ℹ  Difference note: {diff_str} pts (may reflect dynamic or random joker effects)")


def _action_test_silo3(client: BalatroClient) -> None:
    """Run the Silo 3 live test sequence: full combat decision loop (Play vs Discard)."""
    from bots.combat_solver import CombatAction, decide_combat_action

    print("── Silo 3 live test (Combat Decision Loop) ──────────")

    # 1. Fetch current state & navigate to combat if needed
    state = fetch_gamestate(client)
    if state.state in (GameScreen.MENU, GameScreen.BLIND_SELECT):
        print(f"  Current screen is {state.state}, navigating to combat...")
        state = navigate_to_combat(client)

    turn = 0
    max_turns = 20

    while state.state == GameScreen.SELECTING_HAND and turn < max_turns:
        turn += 1
        active_blind = state.active_blind
        target_score = active_blind.score if active_blind else 300
        chips_needed = max(0, target_score - state.round.chips)

        print(f"\n┌── [Turn {turn}] ──────────────────────────────────────────")
        print(f"│  Blind         : {active_blind.name if active_blind else 'Unknown'} ({active_blind.type if active_blind else ''})")
        print(f"│  Chips         : {state.round.chips:,} / {target_score:,} (Needed: {chips_needed:,})")
        print(f"│  Hands left    : {state.round.hands_left} | Discards left: {state.round.discards_left}")
        print(f"│  Cards in hand : {state.hand.count}")

        # Compute decision
        decision = decide_combat_action(state)
        action_upper = decision.action.upper()
        card_labels = []
        for c in decision.cards:
            r = c.value.rank or ""
            s = c.value.suit or ""
            base = f"{r}{s}" if (r or s) else (c.label or "Card")
            extras = []
            if c.modifier.enhancement:
                extras.append(c.modifier.enhancement)
            if c.modifier.edition:
                extras.append(c.modifier.edition)
            if c.modifier.seal:
                extras.append(f"{c.modifier.seal} Seal")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            card_labels.append(f"{base}{extra_str}")
        cards_str = ", ".join(card_labels) if card_labels else "None"

        print(f"│  Decision      : {action_upper}")
        print(f"│  Cards ({len(decision.indices)})    : {cards_str}")
        print(f"│  Reason        : {decision.reason}")
        if decision.action == CombatAction.PLAY:
            print(f"│  Predicted pts : {decision.predicted_score:,}")
        print("└─────────────────────────────────────────────────────")

        # Execute decision
        old_chips = state.round.chips
        raw_new = client.call(decision.action, {"cards": decision.indices})
        state = parse_gamestate(raw_new)

        if decision.action == CombatAction.PLAY:
            if state.round.chips > old_chips:
                scored = state.round.chips - old_chips
            elif state.active_blind and state.active_blind.score > old_chips and state.state in (GameScreen.ROUND_EVAL, "CASH_OUT", GameScreen.SHOP):
                scored = state.active_blind.score - old_chips
            else:
                scored = max(0, state.round.chips - old_chips)

            diff = scored - decision.predicted_score
            if diff == 0:
                print(f"  > Play scored  : +{scored:,} chips (Total: {state.round.chips:,}) | Score check: EXACT MATCH ✓")
            else:
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                pct_err = (abs(diff) / max(1, scored)) * 100.0
                print(f"  > Play scored  : +{scored:,} chips (Total: {state.round.chips:,}) | Score check: DIFF Δ={diff_str} (Predicted: {decision.predicted_score:,}, Err: {pct_err:.1f}%)")

        # Check combat terminal screens
        if state.state in (GameScreen.ROUND_EVAL, GameScreen.SHOP) or state.state == "CASH_OUT":
            print(f"\n✓ Combat completed successfully! Screen reached: {state.state}")
            break
        elif state.state == GameScreen.GAME_OVER:
            print(f"\n✗ Game Over reached: {state.state}")
            break

    if turn >= max_turns:
        print("\n⚠ Reached max turn safety limit.")


def _action_test_silo4(client: BalatroClient) -> None:
    """Run the Silo 4 live test sequence: cash out if needed, manage shop, and position jokers."""
    from bots.shop_manager import (
        ShopActionType,
        decide_shop_action,
        evaluate_joker_value,
        evaluate_voucher_value,
        get_interest_cap,
        get_joker_role,
        get_protected_interest_floor,
        get_spendable_budget,
        is_survival_mode,
    )

    print("── Silo 4 live test (Economy & Shop Management) ────")

    state = fetch_gamestate(client)
    print(f"  Initial screen : {state.state}")

    # 1. If at ROUND_EVAL, cash out to enter SHOP
    if state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
        print("  Screen is ROUND_EVAL, cashing out...")
        raw = client.call("cash_out")
        state = parse_gamestate(raw)
        print(f"  Cashed out -> Current screen: {state.state}")

    # 2. If in MENU or BLIND_SELECT or SELECTING_HAND, guide into shop
    if state.state in (GameScreen.MENU, GameScreen.BLIND_SELECT):
        print(f"  Current screen is {state.state}, navigating through combat to reach shop...")
        state = navigate_to_combat(client)

    if state.state == GameScreen.SELECTING_HAND:
        from bots.combat_solver import decide_combat_action
        print("  Playing combat to reach shop...")
        c_turns = 0
        while state.state == GameScreen.SELECTING_HAND and c_turns < 15:
            c_turns += 1
            dec = decide_combat_action(state)
            raw = client.call(dec.action, {"cards": dec.indices})
            state = parse_gamestate(raw)
        if state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
            raw = client.call("cash_out")
            state = parse_gamestate(raw)

    if state.state != GameScreen.SHOP:
        print(f"✗ Cannot test Silo 4: Game screen is {state.state}, expected SHOP.")
        sys.exit(1)

    # 3. Inside the SHOP
    cap = get_interest_cap(state)
    floor = get_protected_interest_floor(state)
    spendable = get_spendable_budget(state)
    survival = is_survival_mode(state)

    print("\n┌── [Shop Status] ──────────────────────────────────────────")
    print(f"│  Money          : ${state.money}")
    print(f"│  Interest Cap   : ${cap} (Max interest +${cap // 5})")
    print(f"│  Protected Floor: ${floor} | Spendable Budget: ${spendable}")
    print(f"│  Survival Mode  : {'ACTIVE (Priority: Survival)' if survival else 'Inactive (Priority: Interest)'}")
    print(f"│  Reroll Cost    : ${state.round.reroll_cost}")
    print(f"│  Joker Slots    : {len(state.jokers.cards)} / {state.jokers.limit or 5}")

    if state.jokers.cards:
        print("│  Owned Jokers   :")
        for idx, j in enumerate(state.jokers.cards):
            role = get_joker_role(j.key)
            val = evaluate_joker_value(j, state)
            print(f"│    #{idx} {j.label or j.key} [{role}] (Score: {val:.1f}, Sell: ${j.cost.sell})")

    if state.shop.cards:
        print("│  Shop Items     :")
        for idx, c in enumerate(state.shop.cards):
            role = get_joker_role(c.key)
            val = evaluate_joker_value(c, state)
            print(f"│    #{idx} {c.label or c.key} [{role}] (${c.cost.buy}) (Score: {val:.1f})")

    if state.vouchers.cards:
        print("│  Vouchers       :")
        for idx, v in enumerate(state.vouchers.cards):
            v_val = evaluate_voucher_value(v.key, state)
            print(f"│    #{idx} {v.label or v.key} (${v.cost.buy or 10}) (Score: {v_val:.1f})")
    print("└───────────────────────────────────────────────────────────")

    # 4. Action loop
    action_count = 0
    max_actions = 10

    while state.state == GameScreen.SHOP and action_count < max_actions:
        action_count += 1
        action = decide_shop_action(state)

        print(f"\n> Step {action_count}: {action.action_type.upper()}")
        print(f"  Target : {action.target_name} | Cost: ${action.cost}")
        print(f"  Reason : {action.reason}")

        if action.action_type == ShopActionType.NEXT_ROUND:
            raw = client.call("next_round")
            state = parse_gamestate(raw)
            print(f"  Left shop -> Current screen: {state.state}")
            break

        # Execute buy / sell / rearrange / reroll
        raw = client.call(action.rpc_method, action.rpc_params)
        state = parse_gamestate(raw)

    print("\n✓ Silo 4 shop execution complete!")
    if state.jokers.cards:
        joker_summary = " -> ".join([j.label or j.key for j in state.jokers.cards])
        print(f"  Final Joker Order: [ {joker_summary} ]")


def _action_test_silo5(client: BalatroClient) -> None:
    """Run the Silo 5 live test sequence: manage consumables, buy boosters, and pick pack cards."""
    from bots.pack_consumable_manager import (
        decide_pack_action,
        evaluate_pack_purchase,
        manage_held_consumables,
    )

    print("── Silo 5 live test (Boosters & Consumables) ────────")

    state = fetch_gamestate(client)
    print(f"  Initial screen : {state.state}")

    # 1. If at ROUND_EVAL, cash out to enter SHOP
    if state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
        print("  Screen is ROUND_EVAL, cashing out...")
        raw = client.call("cash_out")
        state = parse_gamestate(raw)
        print(f"  Cashed out -> Current screen: {state.state}")

    # 2. If at MENU or BLIND_SELECT, navigate into combat
    if state.state in (GameScreen.MENU, GameScreen.BLIND_SELECT):
        print(f"  Current screen is {state.state}, navigating through combat...")
        state = navigate_to_combat(client)

    # 3. If in combat (SELECTING_HAND), check held consumables, then finish combat to enter shop
    if state.state == GameScreen.SELECTING_HAND:
        from bots.combat_solver import decide_combat_action
        c_turns = 0
        while state.state == GameScreen.SELECTING_HAND and c_turns < 15:
            c_turns += 1

            # Check if any held consumable can be used in combat
            held_act = manage_held_consumables(state)
            if held_act and held_act.action_type == "use":
                print(f"\n[Combat Consumable] Using {held_act.card_name}")
                if held_act.target_indices:
                    target_names = [
                        f"{state.hand.cards[i].value.rank}{state.hand.cards[i].value.suit}"
                        for i in held_act.target_indices
                        if i < len(state.hand.cards)
                    ]
                    print(f"  Target cards   : {', '.join(target_names)} (Indices: {held_act.target_indices})")
                print(f"  Reason         : {held_act.reason}")
                raw = client.call(held_act.rpc_method, held_act.rpc_params)
                state = parse_gamestate(raw)
                continue

            dec = decide_combat_action(state)
            raw = client.call(dec.action, {"cards": dec.indices})
            state = parse_gamestate(raw)

        if state.state in (GameScreen.ROUND_EVAL, "CASH_OUT"):
            raw = client.call("cash_out")
            state = parse_gamestate(raw)

    # 4. If in SHOP: use any shop-usable consumables, then evaluate and buy a Booster Pack
    if state.state == GameScreen.SHOP:
        print("\n┌── [Shop Consumables & Boosters] ──────────────────────────")
        print(f"│  Money          : ${state.money}")
        print(f"│  Consumables    : {len(state.consumables.cards)} / {state.consumables.limit or 2}")
        for idx, c in enumerate(state.consumables.cards):
            print(f"│    #{idx} {c.label or c.key} ({c.set})")

        # Check held consumables in shop first (e.g. Planets or 0-target tarots)
        held_act = manage_held_consumables(state)
        if held_act:
            print(f"│  > Executing held consumable: {held_act.action_type.upper()} {held_act.card_name}")
            print(f"│    Reason: {held_act.reason}")
            raw = client.call(held_act.rpc_method, held_act.rpc_params)
            state = parse_gamestate(raw)

        print(f"│  Booster Packs Available ({len(state.packs.cards)}):")
        for idx, p in enumerate(state.packs.cards):
            p_val = evaluate_pack_purchase(p, state)
            print(f"│    #{idx} {p.label or p.key} (${p.cost.buy}) (Score: {p_val:.1f})")
        print("└───────────────────────────────────────────────────────────")

        # Pick best booster pack to purchase
        if state.packs.cards:
            scored_packs = [
                (idx, p, evaluate_pack_purchase(p, state))
                for idx, p in enumerate(state.packs.cards)
            ]
            scored_packs.sort(key=lambda item: item[2], reverse=True)
            best_idx, best_pack, best_score = scored_packs[0]

            if best_score > 0 and state.money >= (best_pack.cost.buy or 4):
                print(f"\n> Buying Booster Pack: {best_pack.label or best_pack.key} (Score: {best_score:.1f}, Cost: ${best_pack.cost.buy})")
                raw = client.call("buy", {"pack": best_idx})
                state = parse_gamestate(raw)
                print(f"  Pack opened -> Current screen: {state.state}")

    # 5. Handle Opened Booster Pack (SMODS_BOOSTER_OPENED or specific pack states)
    pack_choices = 0
    max_pack_choices = 5

    while state.state in GameScreen.BOOSTER_STATES and pack_choices < max_pack_choices:
        pack_choices += 1
        pack_cards = state.pack.cards
        print(f"\n┌── [Opened Booster Pack - Selection {pack_choices}] ─────────────────")
        print(f"│  Cards inside pack ({len(pack_cards)}):")
        for idx, c in enumerate(pack_cards):
            c_name = c.label or c.key
            if c.value.rank or c.value.suit:
                c_name = f"{c.value.rank}{c.value.suit}"
            mods = []
            if c.modifier.enhancement:
                mods.append(c.modifier.enhancement)
            if c.modifier.edition:
                mods.append(c.modifier.edition)
            if c.modifier.seal:
                mods.append(f"{c.modifier.seal} Seal")
            mod_str = f" [{', '.join(mods)}]" if mods else ""
            print(f"│    #{idx} {c_name} ({c.set}){mod_str}")

        if state.hand.cards:
            hand_summary = ", ".join([f"{c.value.rank}{c.value.suit}" for c in state.hand.cards])
            print(f"│  Cards in hand: [ {hand_summary} ]")

        action = decide_pack_action(state)
        print(f"│  Decision      : {action.action_type.upper()}")
        print(f"│  Selected Card : {action.card_name} (Index: {action.card_index})")
        if action.target_indices:
            target_str = ", ".join([f"{state.hand.cards[i].value.rank}{state.hand.cards[i].value.suit}" for i in action.target_indices if i < len(state.hand.cards)])
            print(f"│  Target Cards  : {target_str} (Indices: {action.target_indices})")
        print(f"│  Score         : {action.score:.1f}")
        print(f"│  Reason        : {action.reason}")
        print("└───────────────────────────────────────────────────────────")

        # Execute pack selection or skip
        raw = client.call(action.rpc_method, action.rpc_params)
        state = parse_gamestate(raw)
        print(f"  Action executed -> Current screen: {state.state}")

    print("\n✓ Silo 5 Booster & Consumables execution complete!")


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.stats:
        from bots.telemetry import print_summary
        print_summary()
        return

    with _make_client(args) as client:
        try:
            if args.dump_state:
                _action_dump_state(client, args.dump_state)
            elif args.test_silo1:
                _action_test_silo1(client)
            elif args.test_silo2:
                _action_test_silo2(client)
            elif args.test_silo3:
                _action_test_silo3(client)
            elif args.test_silo4:
                _action_test_silo4(client)
            elif args.test_silo5:
                _action_test_silo5(client)
            elif args.agent == "rl":
                from bots.rl.eval import evaluate_rl_agent
                evaluate_rl_agent(
                    model_path=args.model_path,
                    client=client,
                    num_runs=args.runs,
                    port=args.port,
                    delay=0.0 if args.fast else args.delay,
                    seed=args.seed,
                    endless=args.endless,
                )
            else:
                from bots.runner import BotRunner
                runner = BotRunner(
                    client=client,
                    endless=args.endless,
                    delay=0.0 if args.fast else args.delay,
                    seed=args.seed,
                )
                runner.run_session(num_runs=args.runs)
        except RPCError as exc:
            print(f"✗ RPC error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
