"""Deterministic RL constants and vocabulary registries for Balatro Macro-RL.
"""

from __future__ import annotations

# ── Fixed Dimensions & Ceilings ─────────────────────────────────────────
MAX_JOKERS: int = 10
MAX_CONSUMABLES: int = 4
MAX_SHOP_JOKERS: int = 4
MAX_SHOP_BOOSTERS: int = 2
MAX_PACK_CHOICES: int = 5
MAX_OFFERS: int = 12

D_CTX: int = 45
D_ENTITY: int = 30
D_DECK: int = 32
D_HAND: int = 24

# ── Symbolic Enums ───────────────────────────────────────────────────────
class HandType:
    HIGH_CARD = "High Card"
    PAIR = "Pair"
    TWO_PAIR = "Two Pair"
    THREE_OF_A_KIND = "Three of a Kind"
    STRAIGHT = "Straight"
    FLUSH = "Flush"
    FULL_HOUSE = "Full House"
    FOUR_OF_A_KIND = "Four of a Kind"
    STRAIGHT_FLUSH = "Straight Flush"
    FIVE_OF_A_KIND = "Five of a Kind"
    FLUSH_HOUSE = "Flush House"
    FLUSH_FIVE = "Flush Five"

class Edition:
    NONE = "NONE"
    FOIL = "FOIL"
    HOLO = "HOLO"
    POLYCHROME = "POLYCHROME"
    NEGATIVE = "NEGATIVE"

class Enhancement:
    NONE = "NONE"
    BONUS = "BONUS"
    MULT = "MULT"
    WILD = "WILD"
    GLASS = "GLASS"
    STEEL = "STEEL"
    STONE = "STONE"
    GOLD = "GOLD"
    LUCKY = "LUCKY"

class Seal:
    NONE = "NONE"
    RED = "RED"
    BLUE = "BLUE"
    GOLD = "GOLD"
    PURPLE = "PURPLE"

class JokerRole:
    NONE = "NONE"
    ECONOMY = "ECONOMY"
    CHIPS = "CHIPS"
    ADD_MULT = "ADD_MULT"
    X_MULT = "X_MULT"
    RETRIGGER_COPY = "RETRIGGER_COPY"
    RULE_UTILITY = "RULE_UTILITY"

# ── JOKER ─────────────────────────────────────────────────────────
JOKER_LIST: tuple[str, ...] = (
    "NONE",
    "j_joker",
    "j_greedy_joker",
    "j_lusty_joker",
    "j_wrathful_joker",
    "j_gluttenous_joker",
    "j_jolly",
    "j_zany",
    "j_mad",
    "j_crazy",
    "j_droll",
    "j_sly",
    "j_wily",
    "j_clever",
    "j_devious",
    "j_crafty",
    "j_half",
    "j_stencil",
    "j_four_fingers",
    "j_mime",
    "j_credit_card",
    "j_ceremonial",
    "j_banner",
    "j_mystic_summit",
    "j_marble",
    "j_loyalty_card",
    "j_8_ball",
    "j_misprint",
    "j_dusk",
    "j_raised_fist",
    "j_chaos",
    "j_fibonacci",
    "j_steel_joker",
    "j_scary_face",
    "j_abstract",
    "j_delayed_grat",
    "j_hack",
    "j_pareidolia",
    "j_gros_michel",
    "j_even_steven",
    "j_odd_todd",
    "j_scholar",
    "j_business",
    "j_supernova",
    "j_ride_the_bus",
    "j_space",
    "j_egg",
    "j_burglar",
    "j_blackboard",
    "j_runner",
    "j_ice_cream",
    "j_dna",
    "j_splash",
    "j_blue_joker",
    "j_sixth_sense",
    "j_constellation",
    "j_hiker",
    "j_faceless",
    "j_green_joker",
    "j_superposition",
    "j_todo_list",
    "j_cavendish",
    "j_card_sharp",
    "j_red_card",
    "j_madness",
    "j_square",
    "j_seance",
    "j_riff_raff",
    "j_vampire",
    "j_shortcut",
    "j_hologram",
    "j_vagabond",
    "j_baron",
    "j_cloud_9",
    "j_rocket",
    "j_obelisk",
    "j_midas_mask",
    "j_luchador",
    "j_photograph",
    "j_gift",
    "j_turtle_bean",
    "j_erosion",
    "j_reserved_parking",
    "j_mail",
    "j_to_the_moon",
    "j_hallucination",
    "j_fortune_teller",
    "j_juggler",
    "j_drunkard",
    "j_stone",
    "j_golden",
    "j_lucky_cat",
    "j_baseball",
    "j_bull",
    "j_diet_cola",
    "j_trading",
    "j_flash",
    "j_popcorn",
    "j_trousers",
    "j_ancient",
    "j_ramen",
    "j_walkie_talkie",
    "j_selzer",
    "j_castle",
    "j_smiley",
    "j_campfire",
    "j_ticket",
    "j_mr_bones",
    "j_acrobat",
    "j_sock_and_buskin",
    "j_swashbuckler",
    "j_troubadour",
    "j_certificate",
    "j_smeared",
    "j_throwback",
    "j_hanging_chad",
    "j_rough_gem",
    "j_bloodstone",
    "j_arrowhead",
    "j_onyx_agate",
    "j_glass",
    "j_ring_master",
    "j_flower_pot",
    "j_blueprint",
    "j_wee",
    "j_merry_andy",
    "j_oops",
    "j_idol",
    "j_seeing_double",
    "j_matador",
    "j_hit_the_road",
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
    "j_stuntman",
    "j_invisible",
    "j_brainstorm",
    "j_satellite",
    "j_shoot_the_moon",
    "j_drivers_license",
    "j_cartomancer",
    "j_astronomer",
    "j_burnt",
    "j_bootstraps",
    "j_caino",
    "j_triboulet",
    "j_yorick",
    "j_chicot",
    "j_perkeo",
)

JOKER_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(JOKER_LIST)}
IDX_TO_JOKER: dict[int, str] = {i: k for i, k in enumerate(JOKER_LIST)}
JOKER_TO_INDEX = JOKER_TO_IDX
INDEX_TO_JOKER = IDX_TO_JOKER
JOKER_INDEX = JOKER_TO_IDX

# ── TAROT ─────────────────────────────────────────────────────────
TAROT_LIST: tuple[str, ...] = (
    "NONE",
    "c_fool",
    "c_magician",
    "c_high_priestess",
    "c_empress",
    "c_emperor",
    "c_heirophant",
    "c_lovers",
    "c_chariot",
    "c_justice",
    "c_hermit",
    "c_wheel_of_fortune",
    "c_strength",
    "c_hanged_man",
    "c_death",
    "c_temperance",
    "c_devil",
    "c_tower",
    "c_star",
    "c_moon",
    "c_sun",
    "c_judgement",
    "c_world",
)

TAROT_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(TAROT_LIST)}
IDX_TO_TAROT: dict[int, str] = {i: k for i, k in enumerate(TAROT_LIST)}
TAROT_TO_INDEX = TAROT_TO_IDX
INDEX_TO_TAROT = IDX_TO_TAROT
TAROT_INDEX = TAROT_TO_IDX

# ── PLANET ─────────────────────────────────────────────────────────
PLANET_LIST: tuple[str, ...] = (
    "NONE",
    "c_mercury",
    "c_venus",
    "c_earth",
    "c_mars",
    "c_jupiter",
    "c_saturn",
    "c_uranus",
    "c_neptune",
    "c_pluto",
    "c_planet_x",
    "c_ceres",
    "c_eris",
)

PLANET_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(PLANET_LIST)}
IDX_TO_PLANET: dict[int, str] = {i: k for i, k in enumerate(PLANET_LIST)}
PLANET_TO_INDEX = PLANET_TO_IDX
INDEX_TO_PLANET = IDX_TO_PLANET
PLANET_INDEX = PLANET_TO_IDX

# ── SPECTRAL ─────────────────────────────────────────────────────────
SPECTRAL_LIST: tuple[str, ...] = (
    "NONE",
    "c_familiar",
    "c_grim",
    "c_incantation",
    "c_talisman",
    "c_aura",
    "c_wraith",
    "c_sigil",
    "c_ouija",
    "c_ectoplasm",
    "c_immolate",
    "c_ankh",
    "c_deja_vu",
    "c_hex",
    "c_trance",
    "c_medium",
    "c_cryptid",
    "c_soul",
    "c_black_hole",
)

SPECTRAL_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(SPECTRAL_LIST)}
IDX_TO_SPECTRAL: dict[int, str] = {i: k for i, k in enumerate(SPECTRAL_LIST)}
SPECTRAL_TO_INDEX = SPECTRAL_TO_IDX
INDEX_TO_SPECTRAL = IDX_TO_SPECTRAL
SPECTRAL_INDEX = SPECTRAL_TO_IDX

# ── VOUCHER ─────────────────────────────────────────────────────────
VOUCHER_LIST: tuple[str, ...] = (
    "NONE",
    "v_overstock_norm",
    "v_clearance_sale",
    "v_hone",
    "v_reroll_surplus",
    "v_crystal_ball",
    "v_telescope",
    "v_grabber",
    "v_wasteful",
    "v_tarot_merchant",
    "v_planet_merchant",
    "v_seed_money",
    "v_blank",
    "v_magic_trick",
    "v_hieroglyph",
    "v_directors_cut",
    "v_paint_brush",
    "v_overstock_plus",
    "v_liquidation",
    "v_glow_up",
    "v_reroll_glut",
    "v_omen_globe",
    "v_observatory",
    "v_nacho_tong",
    "v_recyclomancy",
    "v_tarot_tycoon",
    "v_planet_tycoon",
    "v_money_tree",
    "v_antimatter",
    "v_illusion",
    "v_petroglyph",
    "v_retcon",
    "v_palette",
)

VOUCHER_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(VOUCHER_LIST)}
IDX_TO_VOUCHER: dict[int, str] = {i: k for i, k in enumerate(VOUCHER_LIST)}
VOUCHER_TO_INDEX = VOUCHER_TO_IDX
INDEX_TO_VOUCHER = IDX_TO_VOUCHER
VOUCHER_INDEX = VOUCHER_TO_IDX

# ── BOOSTER ─────────────────────────────────────────────────────────
BOOSTER_LIST: tuple[str, ...] = (
    "NONE",
    "p_arcana_normal_1",
    "p_arcana_normal_2",
    "p_arcana_normal_3",
    "p_arcana_normal_4",
    "p_arcana_jumbo_1",
    "p_arcana_jumbo_2",
    "p_arcana_mega_1",
    "p_arcana_mega_2",
    "p_celestial_normal_1",
    "p_celestial_normal_2",
    "p_celestial_normal_3",
    "p_celestial_normal_4",
    "p_celestial_jumbo_1",
    "p_celestial_jumbo_2",
    "p_celestial_mega_1",
    "p_celestial_mega_2",
    "p_spectral_normal_1",
    "p_spectral_normal_2",
    "p_spectral_jumbo_1",
    "p_spectral_mega_1",
    "p_standard_normal_1",
    "p_standard_normal_2",
    "p_standard_normal_3",
    "p_standard_normal_4",
    "p_standard_jumbo_1",
    "p_standard_jumbo_2",
    "p_standard_mega_1",
    "p_standard_mega_2",
    "p_buffoon_normal_1",
    "p_buffoon_normal_2",
    "p_buffoon_jumbo_1",
    "p_buffoon_mega_1",
)

BOOSTER_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(BOOSTER_LIST)}
IDX_TO_BOOSTER: dict[int, str] = {i: k for i, k in enumerate(BOOSTER_LIST)}
BOOSTER_TO_INDEX = BOOSTER_TO_IDX
INDEX_TO_BOOSTER = IDX_TO_BOOSTER
BOOSTER_INDEX = BOOSTER_TO_IDX

# ── BOSS ─────────────────────────────────────────────────────────
BOSS_LIST: tuple[str, ...] = (
    "NONE",
    "The Arm",
    "The Club",
    "The Eye",
    "The Fish",
    "The Flint",
    "The Goad",
    "The Head",
    "The Hook",
    "The House",
    "The Manacle",
    "The Mark",
    "The Mouth",
    "The Needle",
    "The Ox",
    "The Pillar",
    "The Plant",
    "The Psychic",
    "The Serpent",
    "The Tooth",
    "The Wall",
    "The Water",
    "The Wheel",
    "The Window",
    "Amber Acorn",
    "Cerulean Bell",
    "Crimson Heart",
    "Verdant Leaf",
    "Violet Vessel",
)

BOSS_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(BOSS_LIST)}
IDX_TO_BOSS: dict[int, str] = {i: k for i, k in enumerate(BOSS_LIST)}
BOSS_TO_INDEX = BOSS_TO_IDX
INDEX_TO_BOSS = IDX_TO_BOSS
BOSS_INDEX = BOSS_TO_IDX

# ── TAG ─────────────────────────────────────────────────────────
TAG_LIST: tuple[str, ...] = (
    "NONE",
    "Uncommon Tag",
    "Rare Tag",
    "Negative Tag",
    "Foil Tag",
    "Holographic Tag",
    "Polychrome Tag",
    "Investment Tag",
    "Voucher Tag",
    "Boss Tag",
    "Standard Tag",
    "Charm Tag",
    "Meteor Tag",
    "Buffoon Tag",
    "Handy Tag",
    "Garbage Tag",
    "Ethereal Tag",
    "Coupon Tag",
    "Double Tag",
    "Juggle Tag",
    "D6 Tag",
    "Top-up Tag",
    "Speed Tag",
    "Orbital Tag",
    "Economy Tag",
)

TAG_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(TAG_LIST)}
IDX_TO_TAG: dict[int, str] = {i: k for i, k in enumerate(TAG_LIST)}
TAG_TO_INDEX = TAG_TO_IDX
INDEX_TO_TAG = IDX_TO_TAG
TAG_INDEX = TAG_TO_IDX

# ── HAND_TYPE ─────────────────────────────────────────────────────────
HAND_TYPE_LIST: tuple[str, ...] = (
    "NONE",
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Five of a Kind",
    "Flush House",
    "Flush Five",
)

HAND_TYPE_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(HAND_TYPE_LIST)}
IDX_TO_HAND_TYPE: dict[int, str] = {i: k for i, k in enumerate(HAND_TYPE_LIST)}
HAND_TYPE_TO_INDEX = HAND_TYPE_TO_IDX
INDEX_TO_HAND_TYPE = IDX_TO_HAND_TYPE
HAND_TYPE_INDEX = HAND_TYPE_TO_IDX

# ── EDITION ─────────────────────────────────────────────────────────
EDITION_LIST: tuple[str, ...] = (
    "NONE",
    "FOIL",
    "HOLO",
    "POLYCHROME",
    "NEGATIVE",
)

EDITION_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(EDITION_LIST)}
IDX_TO_EDITION: dict[int, str] = {i: k for i, k in enumerate(EDITION_LIST)}
EDITION_TO_INDEX = EDITION_TO_IDX
INDEX_TO_EDITION = IDX_TO_EDITION
EDITION_INDEX = EDITION_TO_IDX

# ── ENHANCEMENT ─────────────────────────────────────────────────────────
ENHANCEMENT_LIST: tuple[str, ...] = (
    "NONE",
    "BONUS",
    "MULT",
    "WILD",
    "GLASS",
    "STEEL",
    "STONE",
    "GOLD",
    "LUCKY",
)

ENHANCEMENT_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(ENHANCEMENT_LIST)}
IDX_TO_ENHANCEMENT: dict[int, str] = {i: k for i, k in enumerate(ENHANCEMENT_LIST)}
ENHANCEMENT_TO_INDEX = ENHANCEMENT_TO_IDX
INDEX_TO_ENHANCEMENT = IDX_TO_ENHANCEMENT
ENHANCEMENT_INDEX = ENHANCEMENT_TO_IDX

# ── SEAL ─────────────────────────────────────────────────────────
SEAL_LIST: tuple[str, ...] = (
    "NONE",
    "RED",
    "BLUE",
    "GOLD",
    "PURPLE",
)

SEAL_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(SEAL_LIST)}
IDX_TO_SEAL: dict[int, str] = {i: k for i, k in enumerate(SEAL_LIST)}
SEAL_TO_INDEX = SEAL_TO_IDX
INDEX_TO_SEAL = IDX_TO_SEAL
SEAL_INDEX = SEAL_TO_IDX

# ── JOKER_ROLE ─────────────────────────────────────────────────────────
JOKER_ROLE_LIST: tuple[str, ...] = (
    "NONE",
    "ECONOMY",
    "CHIPS",
    "ADD_MULT",
    "X_MULT",
    "RETRIGGER_COPY",
    "RULE_UTILITY",
)

JOKER_ROLE_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(JOKER_ROLE_LIST)}
IDX_TO_JOKER_ROLE: dict[int, str] = {i: k for i, k in enumerate(JOKER_ROLE_LIST)}
JOKER_ROLE_TO_INDEX = JOKER_ROLE_TO_IDX
INDEX_TO_JOKER_ROLE = IDX_TO_JOKER_ROLE
JOKER_ROLE_INDEX = JOKER_ROLE_TO_IDX

# ── ITEM_TYPE ─────────────────────────────────────────────────────────
ITEM_TYPE_LIST: tuple[str, ...] = (
    "NONE",
    "JOKER",
    "TAROT",
    "PLANET",
    "SPECTRAL",
    "VOUCHER",
    "BOOSTER",
    "PLAYING_CARD",
)

ITEM_TYPE_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(ITEM_TYPE_LIST)}
IDX_TO_ITEM_TYPE: dict[int, str] = {i: k for i, k in enumerate(ITEM_TYPE_LIST)}
ITEM_TYPE_TO_INDEX = ITEM_TYPE_TO_IDX
INDEX_TO_ITEM_TYPE = IDX_TO_ITEM_TYPE
ITEM_TYPE_INDEX = ITEM_TYPE_TO_IDX

# ── SUIT ─────────────────────────────────────────────────────────
SUIT_LIST: tuple[str, ...] = (
    "NONE",
    "H",
    "D",
    "C",
    "S",
)

SUIT_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(SUIT_LIST)}
IDX_TO_SUIT: dict[int, str] = {i: k for i, k in enumerate(SUIT_LIST)}
SUIT_TO_INDEX = SUIT_TO_IDX
INDEX_TO_SUIT = IDX_TO_SUIT
SUIT_INDEX = SUIT_TO_IDX

# ── RANK ─────────────────────────────────────────────────────────
RANK_LIST: tuple[str, ...] = (
    "NONE",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
    "A",
)

RANK_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(RANK_LIST)}
IDX_TO_RANK: dict[int, str] = {i: k for i, k in enumerate(RANK_LIST)}
RANK_TO_INDEX = RANK_TO_IDX
INDEX_TO_RANK = IDX_TO_RANK
RANK_INDEX = RANK_TO_IDX

# ── GLOBAL_ITEM ─────────────────────────────────────────────────────────
GLOBAL_ITEM_LIST: tuple[str, ...] = (
    "NONE",
    "j_joker",
    "j_greedy_joker",
    "j_lusty_joker",
    "j_wrathful_joker",
    "j_gluttenous_joker",
    "j_jolly",
    "j_zany",
    "j_mad",
    "j_crazy",
    "j_droll",
    "j_sly",
    "j_wily",
    "j_clever",
    "j_devious",
    "j_crafty",
    "j_half",
    "j_stencil",
    "j_four_fingers",
    "j_mime",
    "j_credit_card",
    "j_ceremonial",
    "j_banner",
    "j_mystic_summit",
    "j_marble",
    "j_loyalty_card",
    "j_8_ball",
    "j_misprint",
    "j_dusk",
    "j_raised_fist",
    "j_chaos",
    "j_fibonacci",
    "j_steel_joker",
    "j_scary_face",
    "j_abstract",
    "j_delayed_grat",
    "j_hack",
    "j_pareidolia",
    "j_gros_michel",
    "j_even_steven",
    "j_odd_todd",
    "j_scholar",
    "j_business",
    "j_supernova",
    "j_ride_the_bus",
    "j_space",
    "j_egg",
    "j_burglar",
    "j_blackboard",
    "j_runner",
    "j_ice_cream",
    "j_dna",
    "j_splash",
    "j_blue_joker",
    "j_sixth_sense",
    "j_constellation",
    "j_hiker",
    "j_faceless",
    "j_green_joker",
    "j_superposition",
    "j_todo_list",
    "j_cavendish",
    "j_card_sharp",
    "j_red_card",
    "j_madness",
    "j_square",
    "j_seance",
    "j_riff_raff",
    "j_vampire",
    "j_shortcut",
    "j_hologram",
    "j_vagabond",
    "j_baron",
    "j_cloud_9",
    "j_rocket",
    "j_obelisk",
    "j_midas_mask",
    "j_luchador",
    "j_photograph",
    "j_gift",
    "j_turtle_bean",
    "j_erosion",
    "j_reserved_parking",
    "j_mail",
    "j_to_the_moon",
    "j_hallucination",
    "j_fortune_teller",
    "j_juggler",
    "j_drunkard",
    "j_stone",
    "j_golden",
    "j_lucky_cat",
    "j_baseball",
    "j_bull",
    "j_diet_cola",
    "j_trading",
    "j_flash",
    "j_popcorn",
    "j_trousers",
    "j_ancient",
    "j_ramen",
    "j_walkie_talkie",
    "j_selzer",
    "j_castle",
    "j_smiley",
    "j_campfire",
    "j_ticket",
    "j_mr_bones",
    "j_acrobat",
    "j_sock_and_buskin",
    "j_swashbuckler",
    "j_troubadour",
    "j_certificate",
    "j_smeared",
    "j_throwback",
    "j_hanging_chad",
    "j_rough_gem",
    "j_bloodstone",
    "j_arrowhead",
    "j_onyx_agate",
    "j_glass",
    "j_ring_master",
    "j_flower_pot",
    "j_blueprint",
    "j_wee",
    "j_merry_andy",
    "j_oops",
    "j_idol",
    "j_seeing_double",
    "j_matador",
    "j_hit_the_road",
    "j_duo",
    "j_trio",
    "j_family",
    "j_order",
    "j_tribe",
    "j_stuntman",
    "j_invisible",
    "j_brainstorm",
    "j_satellite",
    "j_shoot_the_moon",
    "j_drivers_license",
    "j_cartomancer",
    "j_astronomer",
    "j_burnt",
    "j_bootstraps",
    "j_caino",
    "j_triboulet",
    "j_yorick",
    "j_chicot",
    "j_perkeo",
    "c_fool",
    "c_magician",
    "c_high_priestess",
    "c_empress",
    "c_emperor",
    "c_heirophant",
    "c_lovers",
    "c_chariot",
    "c_justice",
    "c_hermit",
    "c_wheel_of_fortune",
    "c_strength",
    "c_hanged_man",
    "c_death",
    "c_temperance",
    "c_devil",
    "c_tower",
    "c_star",
    "c_moon",
    "c_sun",
    "c_judgement",
    "c_world",
    "c_mercury",
    "c_venus",
    "c_earth",
    "c_mars",
    "c_jupiter",
    "c_saturn",
    "c_uranus",
    "c_neptune",
    "c_pluto",
    "c_planet_x",
    "c_ceres",
    "c_eris",
    "c_familiar",
    "c_grim",
    "c_incantation",
    "c_talisman",
    "c_aura",
    "c_wraith",
    "c_sigil",
    "c_ouija",
    "c_ectoplasm",
    "c_immolate",
    "c_ankh",
    "c_deja_vu",
    "c_hex",
    "c_trance",
    "c_medium",
    "c_cryptid",
    "c_soul",
    "c_black_hole",
    "v_overstock_norm",
    "v_clearance_sale",
    "v_hone",
    "v_reroll_surplus",
    "v_crystal_ball",
    "v_telescope",
    "v_grabber",
    "v_wasteful",
    "v_tarot_merchant",
    "v_planet_merchant",
    "v_seed_money",
    "v_blank",
    "v_magic_trick",
    "v_hieroglyph",
    "v_directors_cut",
    "v_paint_brush",
    "v_overstock_plus",
    "v_liquidation",
    "v_glow_up",
    "v_reroll_glut",
    "v_omen_globe",
    "v_observatory",
    "v_nacho_tong",
    "v_recyclomancy",
    "v_tarot_tycoon",
    "v_planet_tycoon",
    "v_money_tree",
    "v_antimatter",
    "v_illusion",
    "v_petroglyph",
    "v_retcon",
    "v_palette",
    "p_arcana_normal_1",
    "p_arcana_normal_2",
    "p_arcana_normal_3",
    "p_arcana_normal_4",
    "p_arcana_jumbo_1",
    "p_arcana_jumbo_2",
    "p_arcana_mega_1",
    "p_arcana_mega_2",
    "p_celestial_normal_1",
    "p_celestial_normal_2",
    "p_celestial_normal_3",
    "p_celestial_normal_4",
    "p_celestial_jumbo_1",
    "p_celestial_jumbo_2",
    "p_celestial_mega_1",
    "p_celestial_mega_2",
    "p_spectral_normal_1",
    "p_spectral_normal_2",
    "p_spectral_jumbo_1",
    "p_spectral_mega_1",
    "p_standard_normal_1",
    "p_standard_normal_2",
    "p_standard_normal_3",
    "p_standard_normal_4",
    "p_standard_jumbo_1",
    "p_standard_jumbo_2",
    "p_standard_mega_1",
    "p_standard_mega_2",
    "p_buffoon_normal_1",
    "p_buffoon_normal_2",
    "p_buffoon_jumbo_1",
    "p_buffoon_mega_1",
    "H_2",
    "H_3",
    "H_4",
    "H_5",
    "H_6",
    "H_7",
    "H_8",
    "H_9",
    "H_T",
    "H_J",
    "H_Q",
    "H_K",
    "H_A",
    "D_2",
    "D_3",
    "D_4",
    "D_5",
    "D_6",
    "D_7",
    "D_8",
    "D_9",
    "D_T",
    "D_J",
    "D_Q",
    "D_K",
    "D_A",
    "C_2",
    "C_3",
    "C_4",
    "C_5",
    "C_6",
    "C_7",
    "C_8",
    "C_9",
    "C_T",
    "C_J",
    "C_Q",
    "C_K",
    "C_A",
    "S_2",
    "S_3",
    "S_4",
    "S_5",
    "S_6",
    "S_7",
    "S_8",
    "S_9",
    "S_T",
    "S_J",
    "S_Q",
    "S_K",
    "S_A",
)

GLOBAL_ITEM_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(GLOBAL_ITEM_LIST)}
IDX_TO_GLOBAL_ITEM: dict[int, str] = {i: k for i, k in enumerate(GLOBAL_ITEM_LIST)}
GLOBAL_ITEM_TO_INDEX = GLOBAL_ITEM_TO_IDX
INDEX_TO_GLOBAL_ITEM = IDX_TO_GLOBAL_ITEM
GLOBAL_ITEM_INDEX = GLOBAL_ITEM_TO_IDX

# Convenient aliases
MAX_SHOP_CARDS = MAX_SHOP_JOKERS
