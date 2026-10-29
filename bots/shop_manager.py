"""Economy Engine, Shop Management, and Joker Positioning (Silo 4).

Responsibilities:
- Dynamic Joker and Voucher valuation based on roles, edition, synergy, and Ante scaling.
- Economic policy: Interest floor protection ($25 default, $50 Seed Money, $100 Money Tree)
  versus Survival Mode in early game (Ante 1-2).
- Shop decision loop: buying, selling/upgrading weak jokers, purchasing vouchers, disciplined reroll.
- Left-to-right Joker sorting: Economy/Chips -> +Mult -> xMult, with optimal Blueprint/Brainstorm placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bots.state_parser import Card, ParsedGameState

# ── Roles & Actions ───────────────────────────────────────────────────────

class JokerRole:
    ECONOMY = "ECONOMY"
    CHIPS = "CHIPS"
    ADD_MULT = "ADD_MULT"
    X_MULT = "X_MULT"
    RETRIGGER_COPY = "RETRIGGER_COPY"
    RULE_UTILITY = "RULE_UTILITY"


class ShopActionType:
    BUY_CARD = "buy_card"
    BUY_VOUCHER = "buy_voucher"
    BUY_PACK = "buy_pack"
    USE_CONSUMABLE = "use_consumable"
    SELL_CONSUMABLE = "sell_consumable"
    SELL_JOKER = "sell_joker"
    REARRANGE_JOKERS = "rearrange_jokers"
    REROLL = "reroll"
    NEXT_ROUND = "next_round"


@dataclass
class ShopAction:
    action_type: str  # One of ShopActionType
    rpc_method: str  # "buy", "sell", "rearrange", "reroll", "next_round"
    rpc_params: dict[str, Any]
    target_name: str
    cost: int
    reason: str


# ── Joker Registry (Roles & Baseline Tiers) ───────────────────────────────

# Tuple: (role, base_score, description)
JOKER_REGISTRY: dict[str, tuple[str, float, str]] = {
    # ── RETRIGGER & COPY (S-Tier) ──────────────────────────────────
    "j_blueprint": (JokerRole.RETRIGGER_COPY, 98.0, "Copies ability of Joker to the right"),
    "j_brainstorm": (JokerRole.RETRIGGER_COPY, 96.0, "Copies ability of leftmost Joker"),
    "j_mime": (JokerRole.RETRIGGER_COPY, 85.0, "Retriggers all cards held in hand"),
    "j_sock_and_buskin": (JokerRole.RETRIGGER_COPY, 82.0, "Retriggers all face cards"),
    "j_hanging_chad": (JokerRole.RETRIGGER_COPY, 80.0, "Retriggers first played card 2 times"),
    "j_hack": (JokerRole.RETRIGGER_COPY, 78.0, "Retriggers 2, 3, 4, 5"),
    "j_selzer": (JokerRole.RETRIGGER_COPY, 75.0, "Retriggers all cards for 10 hands"),
    "j_dusk": (JokerRole.RETRIGGER_COPY, 72.0, "Retriggers all cards on final hand"),

    # ── X_MULT (Late-game win conditions) ──────────────────────────
    "j_cavendish": (JokerRole.X_MULT, 92.0, "x3 Mult flat"),
    "j_canio": (JokerRole.X_MULT, 95.0, "x1 Mult when face card destroyed"),
    "j_triboulet": (JokerRole.X_MULT, 95.0, "x2 Mult for Kings and Queens"),
    "j_hologram": (JokerRole.X_MULT, 90.0, "x0.25 Mult per card added to deck"),
    "j_constellation": (JokerRole.X_MULT, 90.0, "x0.1 Mult per Planet used"),
    "j_family": (JokerRole.X_MULT, 90.0, "x4 Mult if Four of a Kind"),
    "j_trio": (JokerRole.X_MULT, 88.0, "x3 Mult if Three of a Kind"),
    "j_baron": (JokerRole.X_MULT, 88.0, "x1.5 Mult per King held in hand"),
    "j_drivers_license": (JokerRole.X_MULT, 85.0, "x3 Mult if 16+ enhanced cards"),
    "j_yorick": (JokerRole.X_MULT, 85.0, "x1 Mult every 23 discards"),
    "j_acrobat": (JokerRole.X_MULT, 84.0, "x3 Mult on final hand of round"),
    "j_ancient": (JokerRole.X_MULT, 84.0, "x1.5 Mult per card of chosen suit"),
    "j_bloodstone": (JokerRole.X_MULT, 84.0, "1/3 chance x1.5 Mult for Hearts"),
    "j_vampire": (JokerRole.X_MULT, 84.0, "x0.1 Mult per enhanced card scored"),
    "j_card_sharp": (JokerRole.X_MULT, 82.0, "x3 Mult if hand already played this round"),
    "j_steel_joker": (JokerRole.X_MULT, 82.0, "x0.2 Mult per Steel card in deck"),
    "j_campfire": (JokerRole.X_MULT, 82.0, "x0.25 Mult per sold card"),
    "j_idol": (JokerRole.X_MULT, 82.0, "x2 Mult per specific rank and suit"),
    "j_duo": (JokerRole.X_MULT, 80.0, "x2 Mult if Pair"),
    "j_order": (JokerRole.X_MULT, 82.0, "x3 Mult if Straight"),
    "j_tribe": (JokerRole.X_MULT, 82.0, "x2 Mult if Flush"),
    "j_blackboard": (JokerRole.X_MULT, 80.0, "x3 Mult if all held cards Spades/Clubs"),
    "j_photograph": (JokerRole.X_MULT, 80.0, "x2 Mult for first face card scored"),
    "j_glass": (JokerRole.X_MULT, 80.0, "x0.75 Mult per destroyed glass card"),
    "j_throwback": (JokerRole.X_MULT, 78.0, "x0.25 Mult per skipped blind"),
    "j_hit_the_road": (JokerRole.X_MULT, 78.0, "x0.5 Mult per Jack discarded"),
    "j_obelisk": (JokerRole.X_MULT, 70.0, "x0.2 Mult consecutive hands not most played"),

    # ── CHIPS (Early & Mid-game stabilization) ─────────────────────
    "j_stuntman": (JokerRole.CHIPS, 85.0, "+250 Chips, -2 hand size"),
    "j_bull": (JokerRole.CHIPS, 78.0, "+2 Chips per dollar held"),
    "j_castle": (JokerRole.CHIPS, 75.0, "+3 Chips per discarded suit card"),
    "j_wee": (JokerRole.CHIPS, 75.0, "+8 Chips when 2 scored"),
    "j_runner": (JokerRole.CHIPS, 74.0, "Straight gains +15 Chips"),
    "j_blue_joker": (JokerRole.CHIPS, 72.0, "+2 Chips per remaining card in deck"),
    "j_banner": (JokerRole.CHIPS, 70.0, "+30 Chips per remaining discard"),
    "j_square": (JokerRole.CHIPS, 70.0, "+4 Chips if hand has exactly 4 cards"),
    "j_hiker": (JokerRole.CHIPS, 70.0, "+5 Chips per card scored"),
    "j_ice_cream": (JokerRole.CHIPS, 68.0, "+100 Chips, -5 per hand played"),
    "j_scary_face": (JokerRole.CHIPS, 68.0, "+30 Chips per face card"),
    "j_arrowhead": (JokerRole.CHIPS, 68.0, "Spades give +50 Chips"),
    "j_odd_todd": (JokerRole.CHIPS, 65.0, "+31 Chips per odd card"),
    "j_walkie_talkie": (JokerRole.CHIPS, 62.0, "10s and 4s give +10 Chips and +4 Mult"),
    "j_crafty": (JokerRole.CHIPS, 55.0, "+80 Chips if Flush"),
    "j_devious": (JokerRole.CHIPS, 55.0, "+100 Chips if Straight"),
    "j_wily": (JokerRole.CHIPS, 55.0, "+100 Chips if Three of a Kind"),
    "j_clever": (JokerRole.CHIPS, 55.0, "+80 Chips if Two Pair"),
    "j_sly": (JokerRole.CHIPS, 50.0, "+50 Chips if Pair"),

    # ── ADD_MULT (Flat +Mult essentials) ───────────────────────────
    "j_bootstraps": (JokerRole.ADD_MULT, 78.0, "+2 Mult per $5 held"),
    "j_fibonacci": (JokerRole.ADD_MULT, 76.0, "+8 Mult for Ace, 2, 3, 5, 8"),
    "j_abstract": (JokerRole.ADD_MULT, 75.0, "+3 Mult per Joker"),
    "j_supernova": (JokerRole.ADD_MULT, 74.0, "+Mult equal to times poker hand played"),
    "j_gros_michel": (JokerRole.ADD_MULT, 72.0, "+15 Mult flat (Cavendish prerequisite)"),
    "j_ride_the_bus": (JokerRole.ADD_MULT, 72.0, "+1 Mult per hand without face cards"),
    "j_onyx_agate": (JokerRole.ADD_MULT, 72.0, "+7 Mult for Clubs"),
    "j_green_joker": (JokerRole.ADD_MULT, 70.0, "+1 Mult per hand, -1 per discard"),
    "j_flash_card": (JokerRole.ADD_MULT, 70.0, "+2 Mult per reroll in shop"),
    "j_swashbuckler": (JokerRole.ADD_MULT, 70.0, "+Mult equal to total sell value"),
    "j_fortune_teller": (JokerRole.ADD_MULT, 70.0, "+1 Mult per Tarot card used"),
    "j_smiley": (JokerRole.ADD_MULT, 68.0, "+5 Mult for face cards"),
    "j_even_steven": (JokerRole.ADD_MULT, 66.0, "+4 Mult for even cards"),
    "j_red_card": (JokerRole.ADD_MULT, 65.0, "+3 Mult per skipped booster pack"),
    "j_half": (JokerRole.ADD_MULT, 65.0, "+20 Mult if hand <= 3 cards"),
    "j_popcorn": (JokerRole.ADD_MULT, 65.0, "+20 Mult, -4 per round"),
    "j_mystic_summit": (JokerRole.ADD_MULT, 60.0, "+15 Mult when 0 discards left"),
    "j_misprint": (JokerRole.ADD_MULT, 60.0, "+0 to +23 Mult"),
    "j_crazy": (JokerRole.ADD_MULT, 55.0, "+12 Mult if Straight"),
    "j_droll": (JokerRole.ADD_MULT, 55.0, "+10 Mult if Flush"),
    "j_zany": (JokerRole.ADD_MULT, 55.0, "+12 Mult if Three of a Kind"),
    "j_mad": (JokerRole.ADD_MULT, 55.0, "+10 Mult if Two Pair"),
    "j_greedy_joker": (JokerRole.ADD_MULT, 55.0, "+3 Mult for Diamonds"),
    "j_lusty_joker": (JokerRole.ADD_MULT, 55.0, "+3 Mult for Hearts"),
    "j_wrathful_joker": (JokerRole.ADD_MULT, 55.0, "+3 Mult for Spades"),
    "j_gluttenous_joker": (JokerRole.ADD_MULT, 55.0, "+3 Mult for Clubs"),
    "j_jolly": (JokerRole.ADD_MULT, 50.0, "+8 Mult if Pair"),
    "j_joker": (JokerRole.ADD_MULT, 45.0, "+4 Mult flat"),

    # ── ECONOMY (Compounding & Bankroll) ───────────────────────────
    "j_trading": (JokerRole.ECONOMY, 85.0, "First discard destroys 1 card and earns $3"),
    "j_to_the_moon": (JokerRole.ECONOMY, 82.0, "Earn extra $1 interest per $5"),
    "j_rocket": (JokerRole.ECONOMY, 78.0, "Earn $1, increases by $2 when Boss beaten"),
    "j_golden": (JokerRole.ECONOMY, 75.0, "+$4 at end of round"),
    "j_mail": (JokerRole.ECONOMY, 72.0, "+$5 per discarded rank"),
    "j_gift": (JokerRole.ECONOMY, 65.0, "+$1 sell value to cards each round"),
    "j_satellite": (JokerRole.ECONOMY, 65.0, "Earn $1 per unique Planet used"),
    "j_egg": (JokerRole.ECONOMY, 60.0, "Gains $3 sell value each round"),
    "j_cloud_9": (JokerRole.ECONOMY, 60.0, "Earn $1 for each 9 in deck"),
    "j_rough_gem": (JokerRole.ECONOMY, 60.0, "Diamonds earn $1"),
    "j_matador": (JokerRole.ECONOMY, 60.0, "Earn $8 when boss ability triggered"),
    "j_delayed_grat": (JokerRole.ECONOMY, 58.0, "Earn $2/discard if no discards used"),
    "j_faceless_joker": (JokerRole.ECONOMY, 55.0, "Earn $5 if 3+ face cards discarded"),
    "j_credit_card": (JokerRole.ECONOMY, 50.0, "Up to -$20 in debt"),

    # ── RULE_UTILITY (Strategic modifiers) ─────────────────────────
    "j_chicot": (JokerRole.RULE_UTILITY, 85.0, "Disables all Boss Blinds"),
    "j_four_fingers": (JokerRole.RULE_UTILITY, 76.0, "Flushes and Straights can be made with 4 cards"),
    "j_shortcut": (JokerRole.RULE_UTILITY, 76.0, "Straights can skip 1 rank"),
    "j_smeared": (JokerRole.RULE_UTILITY, 75.0, "Hearts/Diamonds and Spades/Clubs count as same suit"),
    "j_burglar": (JokerRole.RULE_UTILITY, 72.0, "+3 hands, 0 discards"),
    "j_juggler": (JokerRole.RULE_UTILITY, 70.0, "+1 hand size"),
    "j_oops": (JokerRole.RULE_UTILITY, 70.0, "Doubles all listed probabilities"),
    "j_drunkard": (JokerRole.RULE_UTILITY, 68.0, "+1 discard each round"),
    "j_merry_andy": (JokerRole.RULE_UTILITY, 68.0, "+3 discards, -1 hand size"),
    "j_pareidolia": (JokerRole.RULE_UTILITY, 68.0, "All cards are face cards"),
    "j_luchador": (JokerRole.RULE_UTILITY, 65.0, "Sell to disable current Boss Blind"),
    "j_showman": (JokerRole.RULE_UTILITY, 65.0, "Duplicate cards can appear"),
    "j_splash": (JokerRole.RULE_UTILITY, 58.0, "Every played card counts in scoring"),
}

# Vouchers valuation: key -> (base_value, description)
VOUCHER_VALUES: dict[str, tuple[float, str]] = {
    "v_antimatter": (95.0, "+1 Joker slot"),
    "v_nacho_tong": (90.0, "+1 hand per round"),
    "v_grabber": (88.0, "+1 hand per round"),
    "v_money_tree": (88.0, "Interest cap raised to $100"),
    "v_palette": (88.0, "+1 hand size"),
    "v_seed_money": (85.0, "Interest cap raised to $50"),
    "v_paint_brush": (85.0, "+1 hand size"),
    "v_hieroglyph": (85.0, "-1 Ante, -1 hand"),
    "v_petroglyph": (85.0, "-1 Ante, -1 discard"),
    "v_observatory": (82.0, "Planets in consumable give x1.5 Mult"),
    "v_liquidation": (82.0, "50% discount in shop"),
    "v_retcon": (80.0, "Reroll Boss Blind unlimited"),
    "v_clearance_sale": (80.0, "25% discount in shop"),
    "v_overstock_plus": (80.0, "+1 card slot in shop"),
    "v_overstock_norm": (78.0, "+1 card slot in shop"),
    "v_directors_cut": (78.0, "Reroll Boss Blind once"),
    "v_recyclomancy": (78.0, "+1 discard per round"),
    "v_wasteful": (75.0, "+1 discard per round"),
    "v_glow_up": (75.0, "Special editions appear 4x more often"),
    "v_telescope": (72.0, "Celestial packs contain most played hand"),
    "v_hone": (70.0, "Special editions appear 2x more often"),
    "v_tarot_tycoon": (70.0, "Tarot cards appear 4x more often"),
    "v_tarot_merchant": (68.0, "Tarot cards appear 2x more often"),
    "v_planet_tycoon": (68.0, "Planet cards appear 4x more often"),
    "v_planet_merchant": (65.0, "Planet cards appear 2x more often"),
    "v_magic_trick": (60.0, "Playing cards can appear in shop"),
    "v_illusion": (65.0, "Playing cards can have enhancement/edition/seal"),
}


# ── Helper functions ──────────────────────────────────────────────────────

def get_joker_role(key: str) -> str:
    """Return the role classification for a joker key."""
    if key in JOKER_REGISTRY:
        return JOKER_REGISTRY[key][0]
    return JokerRole.ADD_MULT


def get_interest_cap(state: ParsedGameState) -> int:
    """Return the current interest cap in dollars ($25 default, $50 Seed Money, $100 Money Tree)."""
    raw_vouchers = state.raw.get("used_vouchers", {})
    if isinstance(raw_vouchers, dict):
        if "v_money_tree" in raw_vouchers:
            return 100
        if "v_seed_money" in raw_vouchers:
            return 50
    elif isinstance(raw_vouchers, (list, set)):
        if "v_money_tree" in raw_vouchers:
            return 100
        if "v_seed_money" in raw_vouchers:
            return 50
    return 25


def get_protected_interest_floor(state: ParsedGameState) -> int:
    """Return the protected interest floor in dollars.

    floor = min(floor(money / 5) * 5, interest_cap)
    """
    cap = get_interest_cap(state)
    return min((state.money // 5) * 5, cap)


def is_survival_mode(state: ParsedGameState) -> bool:
    """Return True if the bot must prioritize survival over compounding interest."""
    # Ante 1: Always survival mode (must acquire basic scoring to pass early blinds)
    if state.ante_num <= 1:
        return True
    # Ante 2: Survival mode if player holds 0 scoring jokers (danger of early game over)
    if state.ante_num == 2:
        scoring_jokers = [
            j for j in state.jokers.cards
            if get_joker_role(j.key) in (JokerRole.ADD_MULT, JokerRole.CHIPS, JokerRole.X_MULT)
        ]
        if len(scoring_jokers) == 0:
            return True
    return False


def get_spendable_budget(state: ParsedGameState) -> int:
    """Return the amount of money safe to spend under current economic policy."""
    if is_survival_mode(state):
        return state.money
    floor = get_protected_interest_floor(state)
    return max(0, state.money - floor)


def should_reroll(state: ParsedGameState) -> bool:
    """Determine whether a shop reroll is economically sound and strategically justified."""
    cost = state.round.reroll_cost

    # 1. Free rerolls (e.g. Chaos the Clown) are always taken if available
    if cost == 0:
        return True

    if state.money < cost:
        return False

    cap = get_interest_cap(state)

    # 2. Comfortable bankroll: Money remaining after reroll preserves full interest cap
    # and leaves enough to buy a solid Joker ($30+)
    if (state.money - cost >= cap) and (state.money - cost >= 30):
        return True

    # 3. Emergency reroll in late game (Ante >= 6) if build lacks xMult scaling
    if state.ante_num >= 6:
        xmult_count = sum(
            1 for j in state.jokers.cards
            if get_joker_role(j.key) == JokerRole.X_MULT and not j.state.debuff
        )
        if xmult_count == 0 and state.money >= cost:
            return True

    return False


# ── Dynamic Evaluation ────────────────────────────────────────────────────

def evaluate_joker_value(joker: Card, state: ParsedGameState) -> float:
    """Compute a dynamic utility score for a Joker taking into account:

    - Baseline Tier value from JOKER_REGISTRY
    - Edition bonuses (Negative >> Polychrome > Holo > Foil)
    - Ante progression (Early game favors +Mult/Chips/Econ; Mid/Late favors xMult/Copy)
    - Current Joker build synergy and role balancing
    """
    key = joker.key
    role, base_score, _ = JOKER_REGISTRY.get(key, (JokerRole.ADD_MULT, 50.0, "Standard Joker"))

    # Edition bonus
    edition = (joker.modifier.edition or "").upper()
    edition_bonus = 0.0
    if edition == "NEGATIVE":
        edition_bonus = 50.0  # Takes 0 slots, pure free upside!
    elif edition == "POLYCHROME":
        edition_bonus = 35.0  # Inherent x1.5 Mult
    elif edition == "HOLO":
        edition_bonus = 20.0  # +10 Mult
    elif edition == "FOIL":
        edition_bonus = 12.0  # +50 Chips

    score = base_score + edition_bonus
    ante = state.ante_num

    # 1. Ante Progression Scaling
    if ante <= 2:
        # Early game: flat multipliers and chips carry the run
        if role == JokerRole.ADD_MULT:
            score *= 1.4
        elif role == JokerRole.CHIPS:
            score *= 1.3
        elif role == JokerRole.ECONOMY:
            score *= 1.3
        elif role == JokerRole.X_MULT:
            # xMult is weaker early if we don't have base +Mult to multiply
            owned_add_mult = sum(1 for j in state.jokers.cards if get_joker_role(j.key) == JokerRole.ADD_MULT)
            if owned_add_mult == 0:
                score *= 0.85
    else:
        # Mid & Late game (Ante 3+): exponential xMult and copy jokers are required
        if role == JokerRole.X_MULT:
            score *= 1.5
        elif role == JokerRole.RETRIGGER_COPY:
            score *= 1.5
        elif role == JokerRole.ADD_MULT:
            score *= 0.8
        elif role == JokerRole.CHIPS:
            score *= 0.7
        elif role == JokerRole.ECONOMY:
            score *= 0.4 if ante >= 5 else 0.65

    # 2. Synergy & Build Composition Balancing
    owned_roles = [get_joker_role(j.key) for j in state.jokers.cards if j.id != joker.id]
    chips_count = owned_roles.count(JokerRole.CHIPS)
    add_mult_count = owned_roles.count(JokerRole.ADD_MULT)
    x_mult_count = owned_roles.count(JokerRole.X_MULT)

    # Diminishing returns on too many Chip jokers
    if role == JokerRole.CHIPS and chips_count >= 2:
        score *= 0.6

    # Need at least one solid +Mult source
    if role == JokerRole.ADD_MULT and add_mult_count == 0:
        score *= 1.3

    # Need xMult in mid/late game
    if role == JokerRole.X_MULT and ante >= 3 and x_mult_count == 0:
        score *= 1.4

    return score


def evaluate_voucher_value(key: str, state: ParsedGameState) -> float:
    """Compute the utility score of a voucher."""
    val, _ = VOUCHER_VALUES.get(key, (50.0, "Voucher"))
    # Economy vouchers are extra valuable in early-to-mid game
    if key in ("v_seed_money", "v_money_tree", "v_clearance_sale") and state.ante_num <= 4:
        val += 10.0
    return val


# ── Left-to-Right Joker Positioning ───────────────────────────────────────

def compute_optimal_joker_order(jokers: list[Card], ante: int = 1) -> list[int]:
    """Compute the optimal left-to-right permutation of joker indices.

    Standard Balatro scoring pipeline:
    Left:   ECONOMY, RULE_UTILITY, CHIPS, RETRIGGER_CARDS (trigger before or independent of Mult)
    Center: ADD_MULT (+Mult additives must accumulate before multipliers)
    Right:  X_MULT (multiplies all accumulated Mult)

    Special Jokers:
    - Blueprint: Placed immediately to the LEFT of its best copy target (best xMult in mid/late game, or best +Mult).
    - Brainstorm: Placed on the far RIGHT while ensuring its copy target sits at index 0 (far left).

    Returns:
        A list of 0-based indices from the original `jokers` list representing
        the new optimal order.
    """
    n = len(jokers)
    if n <= 1:
        return list(range(n))

    # Identify copy jokers
    blueprint_indices = [i for i, j in enumerate(jokers) if j.key == "j_blueprint"]
    brainstorm_indices = [i for i, j in enumerate(jokers) if j.key == "j_brainstorm"]
    regular_indices = [i for i in range(n) if i not in blueprint_indices and i not in brainstorm_indices]

    # Helper: Role priority ranking for left-to-right scoring
    def role_sort_rank(idx: int) -> int:
        role = get_joker_role(jokers[idx].key)
        if role in (JokerRole.ECONOMY, JokerRole.RULE_UTILITY, JokerRole.CHIPS, JokerRole.RETRIGGER_COPY):
            return 1
        elif role == JokerRole.ADD_MULT:
            return 2
        elif role == JokerRole.X_MULT:
            return 3
        return 2

    # Sort regular jokers by (role_sort_rank, -estimated_strength)
    sorted_regulars = sorted(regular_indices, key=role_sort_rank)

    # 1. Find best copy target for Blueprint / Brainstorm
    x_mult_targets = [i for i in regular_indices if get_joker_role(jokers[i].key) == JokerRole.X_MULT]
    add_mult_targets = [i for i in regular_indices if get_joker_role(jokers[i].key) == JokerRole.ADD_MULT]
    other_targets = [i for i in regular_indices if i not in x_mult_targets and i not in add_mult_targets]

    if ante >= 3 and x_mult_targets:
        best_target = x_mult_targets[0]
    elif add_mult_targets:
        best_target = add_mult_targets[0]
    elif x_mult_targets:
        best_target = x_mult_targets[0]
    elif other_targets:
        best_target = other_targets[0]
    else:
        best_target = regular_indices[0] if regular_indices else None

    # Handle Brainstorm (copies leftmost card)
    if brainstorm_indices and best_target is not None:
        # Move best_target to the very beginning (index 0) so Brainstorm copies it
        if best_target in sorted_regulars:
            sorted_regulars.remove(best_target)
            sorted_regulars.insert(0, best_target)

    # Handle Blueprint (placed immediately to the left of its target)
    result: list[int] = []
    if blueprint_indices and best_target is not None:
        target_in_regulars = best_target in sorted_regulars
        for idx in sorted_regulars:
            if idx == best_target:
                # Insert all Blueprints immediately before target
                result.extend(blueprint_indices)
            result.append(idx)
        if not target_in_regulars:
            result.extend(blueprint_indices)
    else:
        result = list(sorted_regulars)
        result.extend(blueprint_indices)

    # Place Brainstorms on the far right (so their copied ability triggers at the end)
    result.extend(brainstorm_indices)

    return result


def reorder_jokers(jokers: list[Card], order: list[int]) -> list[Card]:
    """Return a new list of Card objects rearranged according to `order` indices."""
    return [jokers[i] for i in order]


# ── Main Shop Decision Loop ───────────────────────────────────────────────

def decide_shop_action(
    state: ParsedGameState,
    exclude_card_indices: set[int] | None = None,
    exclude_pack_indices: set[int] | None = None,
    exclude_voucher_indices: set[int] | None = None,
) -> ShopAction:
    """Analyze the shop and inventory state to produce the next optimal action.

    Priority:
    1. Rearrange Jokers: Left-to-right alignment and Blueprint positioning.
    2. Buy / Upgrade Jokers: Buy if slot open, or sell weakest if upgrade threshold (+25%) met.
    3. Buy Vouchers: Buy high-impact vouchers if budget allows.
    4. Reroll: Reroll if economical and justified.
    5. Next Round: Advance to blind select when done.
    """
    owned_jokers = state.jokers.cards
    n_jokers = len(owned_jokers)
    max_slots = state.jokers.limit or 5

    # ── Step 0: Manage Held Consumables (Use/Sell in Shop) ─────────────────
    from bots.pack_consumable_manager import (
        evaluate_pack_purchase,
        manage_held_consumables,
    )
    held_action = manage_held_consumables(state)
    if held_action:
        return ShopAction(
            action_type=ShopActionType.USE_CONSUMABLE if held_action.action_type == "use" else ShopActionType.SELL_CONSUMABLE,
            rpc_method=held_action.rpc_method,
            rpc_params=held_action.rpc_params,
            target_name=held_action.card_name,
            cost=0,
            reason=held_action.reason,
        )

    spendable = get_spendable_budget(state)
    survival = is_survival_mode(state)

    # ── Step 0.5: Ultra-High Priority Booster Pack (e.g. Buffoon in Ante 1-3) ──
    for idx, p in enumerate(state.packs.cards):
        if exclude_pack_indices and idx in exclude_pack_indices:
            continue
        p_cost = p.cost.buy or 4
        if state.money >= p_cost:
            p_val = evaluate_pack_purchase(p, state)
            if p_val >= 90.0:
                return ShopAction(
                    action_type=ShopActionType.BUY_PACK,
                    rpc_method="buy",
                    rpc_params={"pack": idx},
                    target_name=p.label or p.key,
                    cost=p_cost,
                    reason=f"Opening High Priority Booster {p.label or p.key} (Score: {p_val:.1f}, Cost: ${p_cost})",
                )

    # ── Step 1: Evaluate Shop Cards (Jokers) ──────────────────────────────

    shop_jokers: list[tuple[int, Card, float]] = []
    for idx, card in enumerate(state.shop.cards):
        # Identify Jokers in shop
        is_joker = card.set.lower() == "joker" or card.key.startswith("j_")
        if is_joker:
            val = evaluate_joker_value(card, state)
            shop_jokers.append((idx, card, val))

    # Sort shop jokers by value descending
    shop_jokers.sort(key=lambda item: item[2], reverse=True)

    # Case A: We have an open slot (strictly n_jokers < max_slots)
    for idx, card, val in shop_jokers:
        if exclude_card_indices and idx in exclude_card_indices:
            continue
        cost = card.cost.buy
        has_slot = n_jokers < max_slots

        if not has_slot:
            continue

        if state.money < cost:
            continue

        # Affordability rules:
        # - Survival mode: spend down to $0 for any decent joker (val >= 50)
        # - Cruise mode: cost <= spendable for regular jokers (val >= 55)
        # - Powerhouse: S-Tier (val >= 85) or Negative joker is bought even if dipping into interest floor
        is_neg = (card.modifier.edition or "").upper() == "NEGATIVE"
        can_buy = (
            survival
            or is_neg
            or val >= 85.0
            or (cost <= spendable and val >= 55.0)
        )

        if can_buy:
            return ShopAction(
                action_type=ShopActionType.BUY_CARD,
                rpc_method="buy",
                rpc_params={"card": idx},
                target_name=card.label or card.key,
                cost=cost,
                reason=f"Buying Joker {card.label or card.key} (Score: {val:.1f}, Cost: ${cost})",
            )

    # Case B: Slots are full (n_jokers >= max_slots) -> Check for Upgrades / Replacements
    if n_jokers >= max_slots and shop_jokers:
        sellable = [
            (i, j, evaluate_joker_value(j, state))
            for i, j in enumerate(owned_jokers)
            if not j.modifier.eternal
        ]
        if sellable:
            sellable.sort(key=lambda item: item[2])
            weakest_idx, weakest_card, weakest_val = sellable[0]
            sell_value = weakest_card.cost.sell

            for idx, card, val in shop_jokers:
                if exclude_card_indices and idx in exclude_card_indices:
                    continue
                cost = card.cost.buy
                total_funds = state.money + sell_value
                if total_funds < cost:
                    continue

                is_neg = (card.modifier.edition or "").upper() == "NEGATIVE"
                # Upgrade threshold: at least 25% better, or high-tier replacing low-tier, or Negative joker
                is_upgrade = is_neg or (val >= weakest_val * 1.25) or (val >= 85.0 and weakest_val < 70.0)
                net_after = total_funds - cost
                econ_ok = survival or is_neg or (val >= 85.0) or (net_after >= get_protected_interest_floor(state))

                if is_upgrade and econ_ok:
                    return ShopAction(
                        action_type=ShopActionType.SELL_JOKER,
                        rpc_method="sell",
                        rpc_params={"joker": weakest_idx},
                        target_name=weakest_card.label or weakest_card.key,
                        cost=-sell_value,
                        reason=(
                            f"Upgrading Joker slot: Selling {weakest_card.label or weakest_card.key} "
                            f"(Value: {weakest_val:.1f}, Sell: +${sell_value}) to buy "
                            f"{card.label or card.key} (Value: {val:.1f}, +{val - weakest_val:.1f})"
                        ),
                    )

    # ── Step 2: Evaluate Vouchers ─────────────────────────────────────────
    for idx, v in enumerate(state.vouchers.cards):
        if exclude_voucher_indices and idx in exclude_voucher_indices:
            continue
        v_cost = v.cost.buy or 10
        if state.money >= v_cost:
            v_val = evaluate_voucher_value(v.key, state)
            is_major_voucher = v_val >= 85.0
            can_buy_voucher = (
                (v_cost <= spendable and v_val >= 70.0)
                or (is_major_voucher and state.money >= v_cost and (state.money - v_cost >= 15 or survival))
            )
            if can_buy_voucher:
                return ShopAction(
                    action_type=ShopActionType.BUY_VOUCHER,
                    rpc_method="buy",
                    rpc_params={"voucher": idx},
                    target_name=v.label or v.key,
                    cost=v_cost,
                    reason=f"Buying Voucher {v.label or v.key} (Score: {v_val:.1f}, Cost: ${v_cost})",
                )

    # ── Step 2.5: Evaluate Booster Packs ──────────────────────────────────
    for idx, p in enumerate(state.packs.cards):
        if exclude_pack_indices and idx in exclude_pack_indices:
            continue
        p_cost = p.cost.buy or 4
        if state.money >= p_cost:
            p_val = evaluate_pack_purchase(p, state)
            if p_val >= 70.0:
                return ShopAction(
                    action_type=ShopActionType.BUY_PACK,
                    rpc_method="buy",
                    rpc_params={"pack": idx},
                    target_name=p.label or p.key,
                    cost=p_cost,
                    reason=f"Opening Booster Pack {p.label or p.key} (Score: {p_val:.1f}, Cost: ${p_cost})",
                )

    # ── Step 3: Reroll Check ──────────────────────────────────────────────
    if should_reroll(state):
        r_cost = state.round.reroll_cost
        return ShopAction(
            action_type=ShopActionType.REROLL,
            rpc_method="reroll",
            rpc_params={},
            target_name="Shop",
            cost=r_cost,
            reason=f"Rerolling shop (Cost: ${r_cost}, Remaining: ${state.money - r_cost})",
        )

    # ── Step 4: Final Joker Rearrangement (Before leaving shop) ────────────
    if n_jokers > 1:
        optimal_order = compute_optimal_joker_order(owned_jokers, state.ante_num)
        if optimal_order != list(range(n_jokers)):
            new_labels = [owned_jokers[i].label or owned_jokers[i].key for i in optimal_order]
            return ShopAction(
                action_type=ShopActionType.REARRANGE_JOKERS,
                rpc_method="rearrange",
                rpc_params={"jokers": optimal_order},
                target_name="Jokers",
                cost=0,
                reason=f"Positioning Jokers left-to-right: {' -> '.join(new_labels)}",
            )

    # ── Step 5: Exit Shop ─────────────────────────────────────────────────
    return ShopAction(
        action_type=ShopActionType.NEXT_ROUND,
        rpc_method="next_round",
        rpc_params={},
        target_name="Blind Select",
        cost=0,
        reason="Shop evaluation complete; advancing to next blind",
    )
