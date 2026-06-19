from sweep.cards import card_from
from sweep.combos import groups_summing_to, maximal_capture_unions


def C(*specs):
    return [card_from(s) for s in specs]


def unions_of(cards, target):
    return maximal_capture_unions(groups_summing_to(cards, target))


def test_groups_basic():
    cards = C("2H", "3H", "4H", "5H")
    groups = {frozenset(g) for g in groups_summing_to(cards, 9)}
    assert groups == {frozenset(C("4H", "5H")), frozenset(C("2H", "3H", "4H"))}


def test_groups_empty_when_nothing_sums():
    assert groups_summing_to(C("2H", "3H"), 9) == []


def test_no_groups_yields_empty_union():
    assert unions_of(C("2H", "3H"), 9) == [frozenset()]


def test_overlapping_groups_give_two_maximal_options():
    # 4+5 and 2+3+4 both make 9 but share the 4: two distinct maximal captures.
    options = unions_of(C("2H", "3H", "4H", "5H"), 9)
    assert sorted(options, key=sorted) == sorted(
        [frozenset(C("4H", "5H")), frozenset(C("2H", "3H", "4H"))], key=sorted
    )


def test_disjoint_groups_must_all_be_taken():
    # 2+7 and 3+6 are disjoint: maximal capture takes all four cards.
    options = unions_of(C("2H", "7H", "3H", "6H"), 9)
    assert options == [frozenset(C("2H", "7H", "3H", "6H"))]


def test_duplicate_values_give_distinct_options():
    # 4+5h or 4+5d: the 4 can only pair once, so two options, one 5 left behind.
    options = unions_of(C("4H", "5H", "5D"), 9)
    assert sorted(options, key=sorted) == sorted(
        [frozenset(C("4H", "5H")), frozenset(C("4H", "5D"))], key=sorted
    )


def test_subset_collections_are_not_maximal():
    # Taking only {4,5} when {2,3,4}... both overlap; but taking nothing or a
    # non-maximal collection must never appear.
    options = unions_of(C("2H", "7H", "3H", "6H"), 9)
    assert frozenset() not in options
    assert frozenset(C("2H", "7H")) not in options
