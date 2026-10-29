"""Silo 1 – Offline unit tests for client_wrapper and state_parser.

Tests cover:
1. JSON-RPC 2.0 request formatting and delay behaviour (client_wrapper)
2. Full parsing of gamestate fixtures (state_parser)
3. Blind-selection decision: always ``select``, never ``skip``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bots.client_wrapper import (
    BAD_REQUEST,
    BalatroClient,
    ClientConfig,
    RPCError,
)
from bots.state_parser import (
    BlindStatus,
    BlindType,
    GameScreen,
    _normalize_edition,
    _parse_card,
    decide_blind_action,
    parse_gamestate,
)

# ── Fixture helpers ───────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a JSON fixture file by basename (without extension)."""
    path = FIXTURES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _make_httpx_response(result: dict[str, Any], request_id: int = 1):
    """Create a mock httpx.Response returning the given *result*."""
    body = {"jsonrpc": "2.0", "result": result, "id": request_id}
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _make_httpx_error_response(code: int, message: str, request_id: int = 1):
    """Create a mock httpx.Response returning a JSON-RPC error."""
    body = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message, "data": {"name": "BAD_REQUEST"}},
        "id": request_id,
    }
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# 1. client_wrapper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestClientWrapper:
    """Tests for :class:`BalatroClient`."""

    def test_call_sends_jsonrpc_format(self):
        """Verify the outgoing payload matches JSON-RPC 2.0 spec."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"status": "ok"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        result = client.call("health")

        # Check outgoing payload shape
        call_args = client._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "health"
        assert "id" in payload
        assert isinstance(payload["id"], int)

        assert result == {"status": "ok"}

    def test_call_with_params(self):
        """Verify params are included in the payload when provided."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"state": "BLIND_SELECT"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        client.call("start", {"deck": "RED", "stake": "WHITE"})

        payload = client._client.post.call_args.kwargs.get("json") or client._client.post.call_args[1]["json"]
        assert payload["params"] == {"deck": "RED", "stake": "WHITE"}

    def test_call_no_params_omits_key(self):
        """When no params are given, the ``params`` key should be absent."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"status": "ok"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        client.call("health")

        payload = client._client.post.call_args.kwargs.get("json") or client._client.post.call_args[1]["json"]
        assert "params" not in payload

    def test_call_increments_request_id(self):
        """Each successive call should use a strictly increasing request ID."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"status": "ok"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        client.call("health")
        id1 = client._client.post.call_args.kwargs.get("json", client._client.post.call_args[1].get("json", {}))["id"]

        client.call("health")
        id2 = client._client.post.call_args.kwargs.get("json", client._client.post.call_args[1].get("json", {}))["id"]

        assert id2 > id1

    def test_call_raises_rpc_error(self):
        """An error response should raise :class:`RPCError`."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_error_response(BAD_REQUEST, "Invalid params")
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        with pytest.raises(RPCError) as exc_info:
            client.call("start", {"deck": "INVALID"})

        assert exc_info.value.code == BAD_REQUEST
        assert "Invalid params" in str(exc_info.value)

    def test_delay_is_respected(self):
        """With a non-zero delay, calls should wait before sending."""
        delay = 0.15
        config = ClientConfig(delay=delay)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"status": "ok"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        # First call: no wait needed (last_call_time == 0)
        client.call("health")
        t0 = time.monotonic()

        # Second call: should wait ~delay
        client.call("health")
        elapsed = time.monotonic() - t0

        # Allow some margin
        assert elapsed >= delay * 0.8, f"Expected ≥{delay * 0.8}s, got {elapsed}s"

    def test_fast_mode_no_delay(self):
        """With delay=0.0 (fast mode), calls should not wait."""
        config = ClientConfig(delay=0.0)
        client = BalatroClient(config)

        mock_resp = _make_httpx_response({"status": "ok"})
        client._client = MagicMock()
        client._client.post.return_value = mock_resp

        t0 = time.monotonic()
        client.call("health")
        client.call("health")
        elapsed = time.monotonic() - t0

        assert elapsed < 0.1, f"Fast mode should be near-instant, got {elapsed}s"

    def test_context_manager(self):
        """Client should work as a context manager and call close() on exit."""
        config = ClientConfig(delay=0.0)
        with BalatroClient(config) as client:
            client._client = MagicMock()
            mock_resp = _make_httpx_response({"status": "ok"})
            client._client.post.return_value = mock_resp
            client.call("health")
        # No exception means __exit__ was called successfully


# ═══════════════════════════════════════════════════════════════════════════
# 2. state_parser – gamestate parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestGamestateParsing:
    """Tests for :func:`parse_gamestate`."""

    def test_parse_blind_select_small(self):
        """Parse a BLIND_SELECT state with Small Blind active."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        assert state.state == GameScreen.BLIND_SELECT
        assert state.deck == "RED"
        assert state.stake == "WHITE"
        assert state.ante_num == 1
        assert state.round_num == 1
        assert state.money == 4
        assert state.won is False
        assert state.seed == "ABCDEF"

    def test_parse_round_info(self):
        """Round information should be correctly extracted."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        assert state.round.hands_left == 4
        assert state.round.hands_played == 0
        assert state.round.discards_left == 4
        assert state.round.discards_used == 0

    def test_parse_blinds(self):
        """All three blinds should be parsed with correct types and statuses."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        assert "small" in state.blinds
        assert "big" in state.blinds
        assert "boss" in state.blinds

        assert state.blinds["small"].type == BlindType.SMALL
        assert state.blinds["small"].status == BlindStatus.SELECT
        assert state.blinds["small"].score == 300

        assert state.blinds["big"].status == BlindStatus.UPCOMING
        assert state.blinds["boss"].status == BlindStatus.UPCOMING
        assert state.blinds["boss"].name == "The Hook"

    def test_parse_active_blind_small(self):
        """active_blind property should return the blind with SELECT status."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        active = state.active_blind
        assert active is not None
        assert active.type == BlindType.SMALL
        assert active.status == BlindStatus.SELECT

    def test_parse_active_blind_big(self):
        raw = _load_fixture("blind_select_big")
        state = parse_gamestate(raw)

        active = state.active_blind
        assert active is not None
        assert active.type == BlindType.BIG

    def test_parse_active_blind_boss(self):
        raw = _load_fixture("blind_select_boss")
        state = parse_gamestate(raw)

        active = state.active_blind
        assert active is not None
        assert active.type == BlindType.BOSS
        assert active.name == "The Window"

    def test_parse_jokers(self):
        """Joker area should be parsed with card details."""
        raw = _load_fixture("blind_select_boss")
        state = parse_gamestate(raw)

        assert state.jokers.count == 2
        assert state.jokers.limit == 5
        assert len(state.jokers.cards) == 2
        assert state.jokers.cards[0].key == "j_joker"
        assert state.jokers.cards[1].key == "j_greedy_joker"
        assert state.jokers.cards[1].modifier.edition == "FOIL"

    def test_parse_hand_cards(self):
        """Hand cards should be parsed with suit, rank, modifiers."""
        raw = _load_fixture("selecting_hand")
        state = parse_gamestate(raw)

        assert state.state == GameScreen.SELECTING_HAND
        assert state.hand.count == 8
        assert len(state.hand.cards) == 8

        ace = state.hand.cards[0]
        assert ace.key == "H_A"
        assert ace.value.suit == "H"
        assert ace.value.rank == "A"
        assert ace.label == "Ace of Hearts"

    def test_parse_hand_levels(self):
        """Poker hand levels should be parsed from the 'hands' dict."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        assert "Pair" in state.hands
        assert state.hands["Pair"].level == 1
        assert state.hands["Pair"].chips == 10
        assert state.hands["Pair"].mult == 2

        assert "Flush" in state.hands
        assert state.hands["Flush"].chips == 35

    def test_parse_menu_state(self):
        """MENU state should parse without errors even with null areas."""
        raw = _load_fixture("menu")
        state = parse_gamestate(raw)

        assert state.state == GameScreen.MENU
        assert state.money == 0
        assert state.jokers.count == 0
        assert state.hand.count == 0

    def test_parse_cards_with_empty_list_modifiers(self):
        """Lua encodes empty tables as [] in JSON; state_parser must handle list modifiers gracefully."""
        raw = {
            "state": "SELECTING_HAND",
            "hand": {
                "count": 1,
                "cards": [
                    {
                        "id": 10,
                        "key": "H_A",
                        "modifier": [],
                        "value": [],
                        "state": [],
                        "cost": [],
                    }
                ],
            },
        }
        state = parse_gamestate(raw)
        assert state.hand.count == 1
        card = state.hand.cards[0]
        assert card.id == 10
        assert card.modifier.seal is None
        assert card.modifier.eternal is False

    def test_raw_preserved(self):
        """The raw dict should be preserved for later inspection."""
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)

        assert state.raw is raw


# ═══════════════════════════════════════════════════════════════════════════
# 3. Blind decision – always SELECT, never SKIP
# ═══════════════════════════════════════════════════════════════════════════


class TestBlindDecision:
    """Guarantee that the bot always plays every blind and never skips."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["blind_select_small", "blind_select_big", "blind_select_boss"],
        ids=["small", "big", "boss"],
    )
    def test_decide_always_select(self, fixture_name: str):
        """For any blind type, decide_blind_action must return 'select'."""
        raw = _load_fixture(fixture_name)
        state = parse_gamestate(raw)
        action = decide_blind_action(state)
        assert action == "select", f"Expected 'select' but got '{action}' for {fixture_name}"

    def test_never_skip_small(self):
        raw = _load_fixture("blind_select_small")
        state = parse_gamestate(raw)
        assert decide_blind_action(state) != "skip"

    def test_never_skip_big(self):
        raw = _load_fixture("blind_select_big")
        state = parse_gamestate(raw)
        assert decide_blind_action(state) != "skip"

    def test_never_skip_boss(self):
        raw = _load_fixture("blind_select_boss")
        state = parse_gamestate(raw)
        assert decide_blind_action(state) != "skip"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Card & Joker Edition Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCardEditionParsing:
    """Ensure all variations of editions (boolean dict, key, e_ prefix, lowercase) parse correctly."""

    def test_normalize_edition_boolean_dict(self):
        assert _normalize_edition({"foil": True}) == "FOIL"
        assert _normalize_edition({"holo": True}) == "HOLO"
        assert _normalize_edition({"polychrome": True}) == "POLYCHROME"
        assert _normalize_edition({"negative": True}) == "NEGATIVE"

    def test_normalize_edition_key_or_type_dict(self):
        assert _normalize_edition({"key": "e_foil"}) == "FOIL"
        assert _normalize_edition({"key": "e_holo"}) == "HOLO"
        assert _normalize_edition({"type": "polychrome"}) == "POLYCHROME"
        assert _normalize_edition({"key": "negative"}) == "NEGATIVE"

    def test_normalize_edition_string(self):
        assert _normalize_edition("FOIL") == "FOIL"
        assert _normalize_edition("foil") == "FOIL"
        assert _normalize_edition("e_foil") == "FOIL"
        assert _normalize_edition("e_holo") == "HOLO"
        assert _normalize_edition("POLYCHROME") == "POLYCHROME"
        assert _normalize_edition("negative") == "NEGATIVE"

    def test_normalize_edition_empty_or_none(self):
        assert _normalize_edition(None) is None
        assert _normalize_edition("") is None
        assert _normalize_edition({}) is None

    def test_parse_card_with_raw_edition_dict(self):
        card_data = {
            "id": 42,
            "key": "j_joker",
            "set": "JOKER",
            "modifier": {"edition": {"foil": True}},
        }
        card = _parse_card(card_data)
        assert card.modifier.edition == "FOIL"

    def test_parse_card_with_ability_extra_and_dynamic_values(self):
        card_data = {
            "id": 10,
            "key": "j_constellation",
            "set": "JOKER",
            "value": {
                "effect": "X1.4 Mult",
                "extra": 1.4,
                "x_mult": 1.4,
                "mult": 0,
                "chips": 0,
            },
        }
        card = _parse_card(card_data)
        assert card.value.extra == 1.4
        assert card.value.x_mult == 1.4
        assert card.value.mult == 0
        assert card.value.chips == 0

