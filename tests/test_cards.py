from sweep.cards import (
    FULL_DECK,
    TOTAL_CARD_POINTS,
    card_from,
    card_points,
    card_str,
    card_value,
)


def test_deck_has_96_points():
    assert TOTAL_CARD_POINTS == 96


def test_spades_worth_face_value():
    assert card_points(card_from("AS")) == 1
    assert card_points(card_from("9S")) == 9
    assert card_points(card_from("KS")) == 13
    assert sum(card_points(c) for c in range(13)) == 91


def test_other_point_cards():
    assert card_points(card_from("AH")) == 1
    assert card_points(card_from("AD")) == 1
    assert card_points(card_from("AC")) == 1
    assert card_points(card_from("10D")) == 2
    assert card_points(card_from("10H")) == 0
    assert card_points(card_from("KD")) == 0


def test_values():
    assert card_value(card_from("AS")) == 1
    assert card_value(card_from("10C")) == 10
    assert card_value(card_from("JH")) == 11
    assert card_value(card_from("QD")) == 12
    assert card_value(card_from("KC")) == 13


def test_str_roundtrip():
    for c in FULL_DECK:
        assert card_from(card_str(c)) == c
