"""Card representation.

A card is an int 0..51, suit-major: card = suit * 13 + rank_index.
Suits: 0=♠ 1=♥ 2=♦ 3=♣. Rank indices: 0=A, 1=2, ..., 9=10, 10=J, 11=Q, 12=K.
Ints keep the engine fast for self-play; use card_str/card_from at the edges.
"""

SUITS = "♠♥♦♣"
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
FULL_DECK = tuple(range(52))

TEN_OF_DIAMONDS = 2 * 13 + 9


def card_suit(card: int) -> int:
    return card // 13


def card_rank(card: int) -> int:
    return card % 13


def card_value(card: int) -> int:
    """Capture value: A=1, 2-10 face, J=11, Q=12, K=13."""
    return card % 13 + 1


def _points(card: int) -> int:
    if card < 13:  # spades
        return card + 1
    if card % 13 == 0:  # non-spade aces
        return 1
    if card == TEN_OF_DIAMONDS:
        return 2
    return 0


CARD_POINTS = tuple(_points(c) for c in FULL_DECK)
TOTAL_CARD_POINTS = sum(CARD_POINTS)  # 96


def card_points(card: int) -> int:
    return CARD_POINTS[card]


def card_str(card: int) -> str:
    return RANKS[card_rank(card)] + SUITS[card_suit(card)]


_SUIT_CHARS = {"S": 0, "H": 1, "D": 2, "C": 3, "♠": 0, "♥": 1, "♦": 2, "♣": 3}


def card_from(spec: str) -> int:
    """Parse '9S', '10♦', 'AH', ... into a card int."""
    spec = spec.strip()
    suit = _SUIT_CHARS[spec[-1].upper()]
    rank = RANKS.index(spec[:-1].upper())
    return suit * 13 + rank
