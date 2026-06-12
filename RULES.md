# Sweep — Rules Specification

This is the authoritative rules spec for this project. The game engine and any AI
training environment must implement exactly these rules. If code and this document
disagree, this document wins (or the discrepancy must be raised and resolved here).

## Overview

- 2 players, standard 52-card deck.
- Card values: A=1, 2–10 = face value, J=11, Q=12, K=13.
- Players capture cards from a shared table to score points. A game is played over
  multiple rounds until one player has a **cumulative lead of at least 200 points**.

## Dealing and round structure

- **Round 1:** the first player is chosen at random.
- **Later rounds:** the player who is *losing* on cumulative point differential deals;
  therefore the player who is *winning* plays first. If the differential is exactly 0,
  the same player keeps dealing — they continue to deal until they take the lead.

**First half deal (in order):**
1. 4 cards face down to the table.
2. 4 cards to the first player (the non-dealer). These 4 cards **must contain at least
   one card of value 9 or higher**; otherwise the deal is redone.
3. 4 cards to the dealer.

**Declaration and opening move:**
- The 4 table cards remain **face down** during the declaration.
- The first player declares **any value 9–13 that they hold** (not necessarily their
  highest).
- The table cards are then revealed, and the first player's opening move must be made
  **at the declared value** — but not necessarily by playing a card of that value
  (e.g., declared 9, then building a 9-pile by playing a 4 onto a table 5 is legal).
  Capturing or building at that value takes priority; throwing the declared card is
  only allowed if no capture/build at that value exists.

**Rest of the deal:**
- After the opening move, each player is dealt 8 more cards (two batches of 4 each),
  bringing each hand to 12 for the first half.
- When both hands are empty, the remaining 24 cards are dealt, 12 to each player
  (second half). No new cards are added to the table.

## Turn actions

On your turn you must play exactly one card from your hand, as one of:

### 1. Pick up (capture)

Play a card and capture table items whose values match it:
- Capture every table card/pile of **exactly equal value**, plus disjoint groups of
  loose cards whose values **sum** to the played card's value.
- **Maximal capture is mandatory.** After a capture at value V, no pile of V and no
  combination of loose table cards summing to V may remain on the table.
- A pile can only ever be captured by a card of exactly its value — it can never be
  absorbed as a *component* of a larger sum.
- Captured cards (and your played card) go to your capture stack. Capturing makes you
  the "last to capture" for end-of-round leftovers.

### 2. Pile on (build)

Play a card combined with table cards to form a pile of value **9–13 only**:
- You may only build a pile of value V if you hold **another** card of value V in hand
  (a guarantee you can capture it).
- At most one pile per value can exist; building at a value absorbs any existing pile
  of that value and all loose-card combinations summing to it (**maximal**, same as
  capture).
- Adding to an existing pile of the same value makes it **doubled** (its card sum is
  2× or more its value). A doubled pile is locked: it can never be raised or altered,
  only captured.
- A non-doubled pile **not created by you** may be raised to a higher value V by
  playing a single hand card such that pile value + card value = V. **The increment
  must come entirely from your hand** — table cards can never be grouped with the pile
  to reach the new value (e.g., raising a 10-pile to 12 with a hand ace plus a table
  ace is illegal). Once raised, any *other* sets of loose table cards summing to V are
  absorbed into the pile (maximality, same as above). You cannot raise a pile you
  created, and you must hold another card of value V.
- Everyone who has built on a pile at its current value counts as a creator of it.

### 3. Throw (discard)

Place your played card face up on the table, capturing nothing:
- A card that can pick something up can **never** be thrown (it may still be used to
  build instead).
- A card that could be added to a pile **you created** can never be thrown.
- **Forbidden** if you created a pile of this card's value and this is your last card
  of that value — that card is reserved to capture your pile (its only legal use is
  capturing).
- Other cards in your hand having capture options does **not** restrict throwing this
  one — the restriction is per card. Throwing a card that *could have* created a new
  pile or raised an opponent's pile is legal.

## Scoring

| Cards | Points |
|---|---|
| Each spade (A♠–K♠) | face value (1–13), 91 total |
| A♥, A♦, A♣ | 1 each |
| 10♦ | 2 |
| Card majority (>26 captured) | +4 (26–26 tie: +2 each) |

Total per round: 100 points (plus sweeps).

**Sweep:** clearing the entire table with a capture scores **+50** — *except* on the
very last play of the round, which never scores a sweep.

**End of round:**
- Any cards left on the table go to the player who captured last.
- Each player's round score = card points + majority bonus + 50 × sweeps.
- The score difference is added to the cumulative point differential.

**End of game:** first player to a cumulative lead of **at least 200** wins.

---

*Spec confirmed and locked on 2026-06-11. Known engine discrepancies at time of
locking: (1) no multi-round game loop / 200-point win check; (2) `causes_sweep` flag
incorrectly true on the round's final play; (3) raises may illegally group table cards
with the pile being raised.*
