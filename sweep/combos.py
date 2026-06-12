"""Capture combinatorics.

The rules require *maximal* capture: when capturing or building at a value V,
every disjoint group of loose table cards summing to V must be taken. Multiple
distinct maximal selections can exist (e.g. two 5s but only one 4 to pair with),
each yielding a distinct legal action.
"""

from .cards import card_value


def groups_summing_to(cards, target):
    """All subsets of `cards` (ints) whose values sum exactly to `target`.

    Returns a list of tuples. Cards are distinct ints, so subsets are unique
    even when values repeat.
    """
    cs = sorted(cards, key=card_value)
    out = []

    def dfs(start, remaining, acc):
        if remaining == 0:
            out.append(tuple(acc))
            return
        for j in range(start, len(cs)):
            v = card_value(cs[j])
            if v > remaining:
                break  # sorted by value, nothing further fits
            acc.append(cs[j])
            dfs(j + 1, remaining - v, acc)
            acc.pop()

    dfs(0, target, [])
    return out


def maximal_capture_unions(groups):
    """Distinct unions of maximal disjoint collections of `groups`.

    A collection is maximal when no group in the full list is disjoint from its
    union (i.e. nothing capturable would be left behind). Returns a sorted list
    of frozensets; [frozenset()] when there are no groups at all.
    """
    if not groups:
        return [frozenset()]
    gsets = [frozenset(g) for g in groups]
    out = set()

    def dfs(start, union):
        extended = False
        for j in range(start, len(gsets)):
            if gsets[j].isdisjoint(union):
                extended = True
                dfs(j + 1, union | gsets[j])
        if not extended and all(not gs.isdisjoint(union) for gs in gsets):
            out.add(union)

    dfs(0, frozenset())
    return sorted(out, key=sorted)
