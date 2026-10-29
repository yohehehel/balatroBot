"""Game-state parser and Silo 1 navigation logic.

Responsibilities:
- Fetch and normalise the ``gamestate`` response from the API.
- Start a new run with Red Deck / White Stake.
- On the BLIND_SELECT screen, always select (play) the active blind — never skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bots.client_wrapper import BalatroClient

# ── Game-state enums (mirrors docs/api.md) ────────────────────────────────

class GameScreen:
    """Constants for the ``state`` field of the gamestate response."""

    MENU = "MENU"
    BLIND_SELECT = "BLIND_SELECT"
    SELECTING_HAND = "SELECTING_HAND"
    ROUND_EVAL = "ROUND_EVAL"
    CASH_OUT = "ROUND_EVAL"  # Convenience alias for ROUND_EVAL screen
    SHOP = "SHOP"
    BOOSTER_OPENED = "SMODS_BOOSTER_OPENED"
    TAROT_PACK = "TAROT_PACK"
    PLANET_PACK = "PLANET_PACK"
    SPECTRAL_PACK = "SPECTRAL_PACK"
    STANDARD_PACK = "STANDARD_PACK"
    BUFFOON_PACK = "BUFFOON_PACK"
    BOOSTER_STATES = {
        "SMODS_BOOSTER_OPENED",
        "TAROT_PACK",
        "PLANET_PACK",
        "SPECTRAL_PACK",
        "STANDARD_PACK",
        "BUFFOON_PACK",
    }
    GAME_OVER = "GAME_OVER"


class BlindType:
    SMALL = "SMALL"
    BIG = "BIG"
    BOSS = "BOSS"


class BlindStatus:
    SELECT = "SELECT"
    CURRENT = "CURRENT"
    UPCOMING = "UPCOMING"
    DEFEATED = "DEFEATED"
    SKIPPED = "SKIPPED"


# ── Lightweight dataclasses for parsed state ──────────────────────────────

@dataclass
class BlindInfo:
    """Parsed information for a single blind (small / big / boss)."""

    type: str
    status: str
    name: str
    effect: str
    score: int
    tag_name: str | None = None
    tag_effect: str | None = None


@dataclass
class RoundInfo:
    hands_left: int = 0
    hands_played: int = 0
    discards_left: int = 0
    discards_used: int = 0
    reroll_cost: int = 5
    chips: int = 0


@dataclass
class CardValue:
    suit: str | None = None
    rank: str | None = None
    effect: str | None = None
    extra: Any = None
    mult: float | None = None
    x_mult: float | None = None
    chips: int | None = None
    perma_bonus: int = 0


@dataclass
class CardModifier:
    seal: str | None = None
    edition: str | None = None
    enhancement: str | None = None
    eternal: bool = False
    perishable: int | None = None
    rental: bool = False


@dataclass
class CardState:
    debuff: bool = False
    hidden: bool = False
    highlight: bool = False


@dataclass
class CardCost:
    sell: int = 0
    buy: int = 0


@dataclass
class Card:
    id: int = 0
    key: str = ""
    set: str = ""
    label: str = ""
    value: CardValue = field(default_factory=CardValue)
    modifier: CardModifier = field(default_factory=CardModifier)
    state: CardState = field(default_factory=CardState)
    cost: CardCost = field(default_factory=CardCost)


@dataclass
class AreaInfo:
    count: int = 0
    limit: int = 0
    highlighted_limit: int = 0
    cards: list[Card] = field(default_factory=list)


@dataclass
class HandLevel:
    """Level information for a single poker hand type."""

    order: int = 0
    level: int = 1
    chips: int = 0
    mult: int = 0
    played: int = 0
    played_this_round: int = 0


@dataclass
class ParsedGameState:
    """Normalised representation of the full gamestate response."""

    state: str = GameScreen.MENU
    round_num: int = 0
    ante_num: int = 0
    money: int = 0
    deck: str = ""
    stake: str = ""
    seed: str = ""
    won: bool = False
    round: RoundInfo = field(default_factory=RoundInfo)
    blinds: dict[str, BlindInfo] = field(default_factory=dict)
    jokers: AreaInfo = field(default_factory=AreaInfo)
    consumables: AreaInfo = field(default_factory=AreaInfo)
    hand: AreaInfo = field(default_factory=AreaInfo)
    hands: dict[str, HandLevel] = field(default_factory=dict)
    shop: AreaInfo = field(default_factory=AreaInfo)
    vouchers: AreaInfo = field(default_factory=AreaInfo)
    packs: AreaInfo = field(default_factory=AreaInfo)
    pack: AreaInfo = field(default_factory=AreaInfo)
    deck_cards: AreaInfo = field(default_factory=AreaInfo)
    consumeable_usage_total: dict[str, int] = field(default_factory=dict)
    last_tarot_planet: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ── Convenience helpers ───────────────────────────────────────────

    @property
    def active_blind(self) -> BlindInfo | None:
        """Return the blind whose status is ``SELECT`` or ``CURRENT``."""
        for blind in self.blinds.values():
            if blind.status in (BlindStatus.SELECT, BlindStatus.CURRENT):
                return blind
        return None


def _normalize_edition(raw_edition: Any) -> str | None:
    if not raw_edition:
        return None
    if isinstance(raw_edition, str):
        clean = raw_edition.strip()
        if clean.lower().startswith("e_"):
            clean = clean[2:]
        return clean.upper() if clean else None
    if isinstance(raw_edition, dict):
        if raw_edition.get("foil"):
            return "FOIL"
        if raw_edition.get("holo"):
            return "HOLO"
        if raw_edition.get("polychrome"):
            return "POLYCHROME"
        if raw_edition.get("negative"):
            return "NEGATIVE"
        k = raw_edition.get("key") or raw_edition.get("type") or ""
        clean = str(k).strip()
        if clean.lower().startswith("e_"):
            clean = clean[2:]
        return clean.upper() if clean else None
    return None


def _parse_card(raw: dict[str, Any]) -> Card:
    val = raw.get("value") if isinstance(raw.get("value"), dict) else {}
    mod = raw.get("modifier") if isinstance(raw.get("modifier"), dict) else {}
    st = raw.get("state") if isinstance(raw.get("state"), dict) else {}
    cst = raw.get("cost") if isinstance(raw.get("cost"), dict) else {}
    return Card(
        id=raw.get("id", 0),
        key=raw.get("key", ""),
        set=raw.get("set", ""),
        label=raw.get("label", ""),
        value=CardValue(
            suit=val.get("suit"),
            rank=val.get("rank"),
            effect=val.get("effect"),
            extra=val.get("extra"),
            mult=val.get("mult"),
            x_mult=val.get("x_mult"),
            chips=val.get("chips"),
            perma_bonus=val.get("perma_bonus", 0),
        ),
        modifier=CardModifier(
            seal=mod.get("seal"),
            edition=_normalize_edition(mod.get("edition")),
            enhancement=mod.get("enhancement"),
            eternal=mod.get("eternal", False),
            perishable=mod.get("perishable"),
            rental=mod.get("rental", False),
        ),
        state=CardState(
            debuff=st.get("debuff", False),
            hidden=st.get("hidden", False),
            highlight=st.get("highlight", False),
        ),
        cost=CardCost(
            sell=cst.get("sell", 0),
            buy=cst.get("buy", 0),
        ),
    )


def _parse_area(raw: dict[str, Any] | None) -> AreaInfo:
    if not isinstance(raw, dict):
        return AreaInfo()
    cards_raw = raw.get("cards", [])
    if not isinstance(cards_raw, list):
        cards_raw = []
    return AreaInfo(
        count=raw.get("count", 0),
        limit=raw.get("limit", 0),
        highlighted_limit=raw.get("highlighted_limit", 0),
        cards=[_parse_card(c) for c in cards_raw if isinstance(c, dict)],
    )


def _parse_blind(raw: dict[str, Any]) -> BlindInfo:
    if not isinstance(raw, dict):
        return BlindInfo(type="", status="", name="", effect="", score=0)
    return BlindInfo(
        type=raw.get("type", ""),
        status=raw.get("status", ""),
        name=raw.get("name", ""),
        effect=raw.get("effect", ""),
        score=raw.get("score", 0),
        tag_name=raw.get("tag_name"),
        tag_effect=raw.get("tag_effect"),
    )


def _parse_round(raw: dict[str, Any] | None) -> RoundInfo:
    if not isinstance(raw, dict):
        return RoundInfo()
    return RoundInfo(
        hands_left=raw.get("hands_left", 0),
        hands_played=raw.get("hands_played", 0),
        discards_left=raw.get("discards_left", 0),
        discards_used=raw.get("discards_used", 0),
        reroll_cost=raw.get("reroll_cost", 5),
        chips=raw.get("chips", 0),
    )


def _parse_hand_levels(raw: dict[str, Any] | None) -> dict[str, HandLevel]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, HandLevel] = {}
    for name, info in raw.items():
        if isinstance(info, dict):
            result[name] = HandLevel(
                order=info.get("order", 0),
                level=info.get("level", 1),
                chips=info.get("chips", 0),
                mult=info.get("mult", 0),
                played=info.get("played", 0),
                played_this_round=info.get("played_this_round", 0),
            )
    return result


def parse_gamestate(raw: dict[str, Any]) -> ParsedGameState:
    """Transform the raw ``gamestate`` response dict into a :class:`ParsedGameState`."""
    blinds_raw = raw.get("blinds") if isinstance(raw.get("blinds"), dict) else {}
    parsed_blinds: dict[str, BlindInfo] = {}
    for key in ("small", "big", "boss"):
        if key in blinds_raw and isinstance(blinds_raw[key], dict):
            parsed_blinds[key] = _parse_blind(blinds_raw[key])

    return ParsedGameState(
        state=raw.get("state", GameScreen.MENU),
        round_num=raw.get("round_num", 0),
        ante_num=raw.get("ante_num", 0),
        money=raw.get("money", 0),
        deck=raw.get("deck", ""),
        stake=raw.get("stake", ""),
        seed=raw.get("seed", ""),
        won=raw.get("won", False),
        round=_parse_round(raw.get("round")),
        blinds=parsed_blinds,
        jokers=_parse_area(raw.get("jokers")),
        consumables=_parse_area(raw.get("consumables")),
        hand=_parse_area(raw.get("hand")),
        hands=_parse_hand_levels(raw.get("hands")),
        shop=_parse_area(raw.get("shop")),
        vouchers=_parse_area(raw.get("vouchers")),
        packs=_parse_area(raw.get("packs")),
        pack=_parse_area(raw.get("pack")),
        deck_cards=_parse_area(raw.get("cards")),
        consumeable_usage_total=raw.get("consumeable_usage_total", {}) if isinstance(raw.get("consumeable_usage_total"), dict) else {},
        last_tarot_planet=raw.get("last_tarot_planet"),
        raw=raw,
    )


# ── Silo 1 Navigation Logic ──────────────────────────────────────────────

def decide_blind_action(state: ParsedGameState) -> str:
    """Return the action to take on the BLIND_SELECT screen.

    Per PLAN_ANTIGRAVITY §2: always play 100 % of blinds — never skip.

    Returns:
        Always ``"select"`` (play the blind).
    """
    return "select"


def fetch_gamestate(client: BalatroClient) -> ParsedGameState:
    """Fetch the current game state from the API and parse it."""
    raw = client.call("gamestate")
    return parse_gamestate(raw)


def start_run(
    client: BalatroClient,
    deck: str = "RED",
    stake: str = "WHITE",
    seed: str | None = None,
) -> ParsedGameState:
    """Start a new run from the MENU screen.

    Always sends ``menu`` first to ensure the menu is fully initialised
    (required by the Lua ``start`` endpoint), then ``start`` with the
    requested deck / stake / seed.
    """
    # Always call menu to ensure full initialisation (mirrors upstream fixtures)
    client.call("menu")

    params: dict[str, Any] = {"deck": deck, "stake": stake}
    if seed:
        params["seed"] = seed

    raw = client.call("start", params)
    return parse_gamestate(raw)


def select_blind(client: BalatroClient) -> ParsedGameState:
    """Select (play) the current blind — never skip.

    Expects the game to be on the ``BLIND_SELECT`` screen.
    """
    raw = client.call("select")
    return parse_gamestate(raw)


def navigate_to_combat(client: BalatroClient) -> ParsedGameState:
    """High-level helper: start a run (if needed) and enter the first combat.

    1. If on MENU → start a Red Deck / White Stake run.
    2. If on BLIND_SELECT → select the blind.
    3. Returns when the state is SELECTING_HAND (in combat).
    """
    state = fetch_gamestate(client)

    if state.state == GameScreen.MENU:
        state = start_run(client)

    if state.state == GameScreen.BLIND_SELECT:
        state = select_blind(client)

    return state
