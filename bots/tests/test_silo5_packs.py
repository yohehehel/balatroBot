"""Unit tests for Silo 5: Booster Packs and Consumables Management.

Tests:
1. Celestial Pack: Face à 3 planètes, le bot choisit la planète correspondant
   à la main la plus jouée (ou Jupiter / Mercury par défaut). Black Hole en priorité absolue.
2. The Hanged Man: Dans un Arcana Pack ou en main, sélectionne les 2 cartes les plus faibles
   hors couleur dominante (en épargnant les cartes Steel, Polychrome, Seals).
3. Death: Sélectionne exactement 2 cartes (la pire et la meilleure) avec la pire carte à gauche
   pour être transformée en la meilleure (et jamais l'inverse).
4. Economic Priority: Avec 20$ en poche, The Hermit (+20$) est préféré à un Tarot mineur.
   Temperance avec Jokers de forte valeur de revente.
5. Spectral Safety: Avec 30$ en poche, refuse Wraith (qui ferait perdre 30$) et skip ou prend autre chose.
   Avec 4 bons Jokers, refuse Ankh et Hex. Ectoplasm refusé si hand size < 8.
6. Inventory Management: Utilisation immédiate des Planètes en inventaire pour libérer le slot.
   Vente des consommables dangereux conservés par erreur.
7. Standard Pack: Refuse les cartes ordinaires pour éviter de diluer le deck, mais prend les cartes
   avec Hologram ou cartes avec Editions/Sceaux.
8. Buffoon Pack: Choisit le meilleur Joker, ou skip si les slots sont pleins.
"""

from __future__ import annotations

from bots.pack_consumable_manager import (
    decide_pack_action,
    evaluate_card_strength,
    evaluate_pack_purchase,
    manage_held_consumables,
    select_targets_for_consumable,
)
from bots.state_parser import (
    AreaInfo,
    Card,
    CardCost,
    CardModifier,
    CardState,
    CardValue,
    HandLevel,
    ParsedGameState,
    RoundInfo,
)

# ── Helpers ───────────────────────────────────────────────────────────────

def _make_card(
    rank: str = "2",
    suit: str = "H",
    enhancement: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    key: str = "",
    c_set: str = "Default",
    buy: int = 2,
    sell: int = 1,
) -> Card:
    return Card(
        key=key or f"{suit}_{rank}",
        set=c_set,
        label=f"{rank} of {suit}",
        value=CardValue(suit=suit, rank=rank),
        modifier=CardModifier(enhancement=enhancement, edition=edition, seal=seal),
        state=CardState(),
        cost=CardCost(buy=buy, sell=sell),
    )


def _make_consumable(key: str, label: str = "", c_set: str = "Tarot") -> Card:
    return Card(
        key=key,
        set=c_set,
        label=label or key,
        value=CardValue(),
        modifier=CardModifier(),
        state=CardState(),
        cost=CardCost(buy=3, sell=1),
    )


def _make_joker(key: str, label: str = "", sell: int = 3, buy: int = 6) -> Card:
    return Card(
        key=key,
        set="Joker",
        label=label or key,
        value=CardValue(),
        modifier=CardModifier(),
        state=CardState(),
        cost=CardCost(buy=buy, sell=sell),
    )


def _make_pack(key: str, label: str = "", buy: int = 6) -> Card:
    return Card(
        key=key,
        set="Booster",
        label=label or key,
        value=CardValue(),
        modifier=CardModifier(),
        state=CardState(),
        cost=CardCost(buy=buy, sell=buy // 2),
    )


def _base_state(money: int = 25, ante: int = 2) -> ParsedGameState:
    return ParsedGameState(
        state="SHOP",
        money=money,
        ante_num=ante,
        round=RoundInfo(reroll_cost=5),
        jokers=AreaInfo(count=2, limit=5, cards=[_make_joker("j_joker", "Joker"), _make_joker("j_sly", "Sly Joker")]),
        consumables=AreaInfo(count=0, limit=2, cards=[]),
        hand=AreaInfo(count=0, limit=8, cards=[]),
        hands={
            "Pair": HandLevel(played=1, level=1),
            "Flush": HandLevel(played=4, level=2),
            "High Card": HandLevel(played=0, level=1),
        },
        packs=AreaInfo(count=0, limit=2, cards=[]),
        pack=AreaInfo(count=0, limit=5, cards=[]),
    )


# ── Test Suite ────────────────────────────────────────────────────────────

class TestSilo5CelestialPack:
    """Test 1: Celestial Pack (Planets) choices."""

    def test_celestial_chooses_most_played_hand_planet(self) -> None:
        """Chooses Jupiter (Flush) because Flush is the most played hand (4 plays)."""
        state = _base_state()
        state.state = "SMODS_BOOSTER_OPENED"
        # Pack contains Mercury (Pair), Jupiter (Flush), Uranus (Two Pair)
        state.pack.cards = [
            _make_consumable("c_mercury", "Mercury", c_set="Planet"),
            _make_consumable("c_jupiter", "Jupiter", c_set="Planet"),
            _make_consumable("c_uranus", "Uranus", c_set="Planet"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.action_type == "select"
        assert action.card_index == 1
        assert action.card_name == "Jupiter"

    def test_celestial_chooses_default_jupiter_or_mercury_when_no_plays(self) -> None:
        """When no hands have been played yet, defaults to reliable Jupiter (Flush) or Mercury (Pair)."""
        state = _base_state()
        state.state = "SMODS_BOOSTER_OPENED"
        state.hands = {
            "Pair": HandLevel(played=0, level=1),
            "Straight": HandLevel(played=0, level=1),
            "Full House": HandLevel(played=0, level=1),
        }
        state.pack.cards = [
            _make_consumable("c_earth", "Earth", c_set="Planet"),
            _make_consumable("c_mercury", "Mercury", c_set="Planet"),
            _make_consumable("c_saturn", "Saturn", c_set="Planet"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_index == 1
        assert action.card_name == "Mercury"

    def test_celestial_black_hole_always_top_priority(self) -> None:
        """Black Hole takes absolute priority over all other planets."""
        state = _base_state()
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_jupiter", "Jupiter", c_set="Planet"),
            _make_consumable("c_black_hole", "Black Hole", c_set="Spectral"),
            _make_consumable("c_mercury", "Mercury", c_set="Planet"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_index == 1
        assert action.score == 100.0


class TestSilo5HangedMan:
    """Test 2: The Hanged Man target selection and protection."""

    def test_hanged_man_destroys_weakest_non_dominant_cards(self) -> None:
        """Selects the 2 lowest rank cards, protecting dominant suit and enhancements."""
        state = _base_state()
        hanged_man = _make_consumable("c_hanged_man", "The Hanged Man")

        # Hand:
        # #0: 2 of Clubs (low, non-dominant, plain) -> Candidate 1
        # #1: 3 of Diamonds (low, non-dominant, plain) -> Candidate 2
        # #2: Ace of Spades (dominant suit)
        # #3: King of Spades (dominant suit)
        # #4: 2 of Spades (Steel card -> PROTECTED!)
        # #5: 4 of Hearts (Polychrome card -> PROTECTED!)
        hand = [
            _make_card("2", "C"),
            _make_card("3", "D"),
            _make_card("A", "S"),
            _make_card("K", "S"),
            _make_card("2", "S", enhancement="Steel"),
            _make_card("4", "H", edition="Polychrome"),
        ]

        targets = select_targets_for_consumable(hanged_man, hand, state)
        assert len(targets) == 2
        assert set(targets) == {0, 1}
        # Verify protected cards were NOT selected
        assert 4 not in targets
        assert 5 not in targets

    def test_hanged_man_spares_sealed_cards(self) -> None:
        """Cards with Red, Blue, Purple, or Gold seals are never destroyed."""
        state = _base_state()
        hanged_man = _make_consumable("c_hanged_man", "The Hanged Man")

        hand = [
            _make_card("2", "C", seal="Red"),
            _make_card("3", "D", seal="Blue"),
            _make_card("5", "H"),
            _make_card("6", "H"),
        ]

        targets = select_targets_for_consumable(hanged_man, hand, state)
        assert len(targets) == 2
        assert set(targets) == {2, 3}


class TestSilo5DeathTargeting:
    """Test 3: Death (duplicate right card over left card)."""

    def test_death_transforms_weaker_left_into_stronger_right(self) -> None:
        """Death selects exactly 2 cards: a weak card at left index transformed into strong card at right index."""
        state = _base_state()
        death = _make_consumable("c_death", "Death")

        # Hand:
        # #0: 2 of Clubs (plain 2 -> weak)
        # #1: 3 of Diamonds (plain 3)
        # #2: Ace of Spades (dominant suit As -> strong)
        # #3: King of Spades (Steel King -> strongest)
        hand = [
            _make_card("2", "C"),
            _make_card("3", "D"),
            _make_card("A", "S"),
            _make_card("K", "S", enhancement="Steel"),
        ]

        targets = select_targets_for_consumable(death, hand, state)
        assert len(targets) == 2
        left_idx, right_idx = targets[0], targets[1]
        assert left_idx < right_idx  # Left card is to the left of right card

        left_strength = evaluate_card_strength(hand[left_idx], state)
        right_strength = evaluate_card_strength(hand[right_idx], state)

        # Ensure the card at left_idx (to be transformed) is strictly weaker than right_idx
        assert left_strength < right_strength
        assert hand[left_idx].value.rank == "2"
        assert hand[right_idx].value.rank == "K"
        assert hand[right_idx].modifier.enhancement == "Steel"


class TestSilo5EconomicPriority:
    """Test 4: Immediate Economy (The Hermit & Temperance)."""

    def test_hermit_prioritized_with_money(self) -> None:
        """With $20 in bank, The Hermit (+20$) is picked over a minor Tarot."""
        state = _base_state(money=20)
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_lovers", "The Lovers"),
            _make_consumable("c_hermit", "The Hermit"),
            _make_consumable("c_tower", "The Tower"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_index == 1
        assert action.card_name == "The Hermit"
        assert action.score >= 90.0

    def test_temperance_prioritized_with_high_joker_sell_value(self) -> None:
        """With Jokers having total sell value of $18, Temperance is highly scored."""
        state = _base_state(money=10)
        state.jokers.cards = [
            _make_joker("j_blueprint", "Blueprint", sell=10),
            _make_joker("j_brainstorm", "Brainstorm", sell=8),
        ]
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_magician", "The Magician"),
            _make_consumable("c_temperance", "Temperance"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_index == 1
        assert action.card_name == "Temperance"
        assert action.score >= 90.0


class TestSilo5SpectralSafety:
    """Test 5: Anti-suicide checks on dangerous Spectral cards."""

    def test_wraith_refused_when_rich(self) -> None:
        """With $30, bot refuses Wraith (which sets money to $0) and skips or picks safe card."""
        state = _base_state(money=30)
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_wraith", "Wraith", c_set="Spectral"),
        ]

        action = decide_pack_action(state)
        assert action.skip
        assert "Wraith refused" in action.reason

    def test_wraith_accepted_when_poor_with_free_slot(self) -> None:
        """With $4 and open Joker slot, Wraith is safe and high value."""
        state = _base_state(money=4)
        state.jokers.cards = [_make_joker("j_joker", "Joker")]
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_wraith", "Wraith", c_set="Spectral"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_name == "Wraith"

    def test_ankh_and_hex_refused_with_multiple_jokers(self) -> None:
        """With 4 Jokers, Ankh and Hex (which destroy all other Jokers) are refused."""
        state = _base_state()
        state.jokers.cards = [
            _make_joker("j_blueprint", "Blueprint"),
            _make_joker("j_cavendish", "Cavendish"),
            _make_joker("j_banner", "Banner"),
            _make_joker("j_joker", "Joker"),
        ]
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_consumable("c_ankh", "Ankh", c_set="Spectral"),
            _make_consumable("c_hex", "Hex", c_set="Spectral"),
        ]

        action = decide_pack_action(state)
        assert action.skip
        assert "refused" in action.reason


class TestSilo5InventoryManagement:
    """Test 6: Held Consumables Inventory Management."""

    def test_immediate_planet_usage_in_inventory(self) -> None:
        """Planets in consumable inventory are used immediately to level up hand and free slot."""
        state = _base_state()
        state.state = "SHOP"
        state.consumables.cards = [
            _make_consumable("c_jupiter", "Jupiter", c_set="Planet"),
        ]

        action = manage_held_consumables(state)
        assert action is not None
        assert action.action_type == "use"
        assert action.consumable_index == 0
        assert action.rpc_method == "use"
        assert "c_jupiter" in action.card_name.lower() or "jupiter" in action.card_name.lower()

    def test_dangerous_consumable_sold(self) -> None:
        """Hex or Ankh held while owning 4 Jokers is sold to avoid accidental trigger."""
        state = _base_state()
        state.jokers.cards = [
            _make_joker("j_1", "Joker 1"),
            _make_joker("j_2", "Joker 2"),
            _make_joker("j_3", "Joker 3"),
        ]
        state.consumables.cards = [
            _make_consumable("c_hex", "Hex", c_set="Spectral"),
        ]

        action = manage_held_consumables(state)
        assert action is not None
        assert action.action_type == "sell"
        assert action.consumable_index == 0


class TestSilo5StandardAndBuffoonPacks:
    """Test 7 & 8: Standard and Buffoon Packs purchasing and choices."""

    def test_standard_pack_skipped_without_hologram_or_edition(self) -> None:
        """Plain playing cards in a Standard pack are skipped to avoid diluting deck."""
        state = _base_state()
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_card("2", "C"),
            _make_card("7", "D"),
            _make_card("9", "H"),
        ]

        action = decide_pack_action(state)
        assert action.skip
        assert "dilution" in action.reason.lower() or "no quality" in action.reason.lower()

    def test_standard_pack_taken_with_hologram(self) -> None:
        """When player owns Hologram, any card added gives +x0.25 Mult, so best card is chosen."""
        state = _base_state()
        state.jokers.cards.append(_make_joker("j_hologram", "Hologram"))
        state.state = "SMODS_BOOSTER_OPENED"
        state.pack.cards = [
            _make_card("2", "C"),
            _make_card("A", "S"),
        ]

        action = decide_pack_action(state)
        assert not action.skip
        assert action.card_index == 1  # Ace of Spades

    def test_buffoon_pack_purchase_priority_in_early_ante(self) -> None:
        """Buffoon pack has very high purchase score (>= 90) in Ante 1-3 with open slots."""
        state = _base_state(money=15, ante=2)
        state.jokers.cards = [_make_joker("j_joker", "Joker")]  # 1/5 slots
        pack = _make_pack("p_buffoon_normal_1", "Buffoon Pack", buy=6)

        score = evaluate_pack_purchase(pack, state)
        assert score >= 90.0
