"""
NRL Squad / team list module.

Fetches player lists from the NRL match centre and computes key_players_out:
  - How many of a team's regular starters in key positions (fullback, halfback,
    five-eighth, hooker) are absent compared to their rolling baseline.

KEY_POSITIONS: jersey numbers that matter most for team performance.
"""

import time
from collections import Counter, defaultdict, deque
from typing import Optional

import requests

NRL_BASE = "https://www.nrl.com"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json",
}

# Jersey numbers for the positions that most influence match outcome
KEY_NUMBERS = {1, 6, 7, 9}      # fullback, five-eighth, halfback, hooker
KEY_WINDOW  = 5                  # look back this many games to define "regular"


# ─── FETCH ───────────────────────────────────────────────────────────────────

def fetch_players(match_centre_url: str) -> dict[str, list[dict]]:
    """
    Hit the match centre data endpoint and return
    {'home': [player, ...], 'away': [player, ...]}
    Each player dict has: {playerId, number, firstName, lastName, position}.
    Returns empty lists if not yet announced.
    """
    url = f"{NRL_BASE}{match_centre_url}data"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {"home": [], "away": []}
        d = r.json()
        def clean(players):
            return [
                {
                    "playerId":  p.get("playerId"),
                    "number":    p.get("number"),
                    "firstName": p.get("firstName", ""),
                    "lastName":  p.get("lastName", ""),
                    "position":  p.get("position", ""),
                }
                for p in players
                if p.get("number") and p.get("playerId")
            ]
        return {
            "home": clean(d.get("homeTeam", {}).get("players", [])),
            "away": clean(d.get("awayTeam", {}).get("players", [])),
        }
    except Exception:
        return {"home": [], "away": []}


# ─── TRACKER ─────────────────────────────────────────────────────────────────

class SquadTracker:
    """
    Maintains a rolling window of starters per team per key position.
    Process games in chronological order.
    """

    def __init__(self, window: int = KEY_WINDOW):
        self._window = window
        # {team: {jersey_number: deque of playerIds}}
        self._history: dict[str, dict[int, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window))
        )

    def regular_starter(self, team: str, number: int) -> Optional[int]:
        """
        Returns the playerId of the regular starter for a key jersey number,
        or None if no clear regular (fewer than 3 appearances in the window).
        """
        history = self._history[team][number]
        if not history:
            return None
        c = Counter(history)
        most_common_id, count = c.most_common(1)[0]
        # Only consider someone "regular" if they've started in 3+ of last N games
        if count >= min(3, len(history)):
            return most_common_id
        return None

    def key_players_out(self, team: str, players: list[dict]) -> int:
        """
        Given the team's player list for THIS game, count how many key positions
        (1, 6, 7, 9) have a non-regular player starting.
        Returns 0 if no history yet (early season) or if all regulars are present.
        """
        if not players:
            return 0
        starter_map = {
            p["number"]: p["playerId"]
            for p in players
            if p["number"] in KEY_NUMBERS
        }
        missing = 0
        for num in KEY_NUMBERS:
            regular = self.regular_starter(team, num)
            if regular is None:
                continue    # no established regular yet — don't penalise
            actual = starter_map.get(num)
            if actual is None or actual != regular:
                missing += 1
        return missing

    def update(self, team: str, players: list[dict]):
        """Record who started in each key position for this game."""
        for p in players:
            if p["number"] in KEY_NUMBERS and p["playerId"]:
                self._history[team][p["number"]].append(p["playerId"])

    def reset_season(self):
        """Clear history at start of a new season."""
        self._history.clear()


def players_out_string(team: str, players: list[dict],
                        tracker: SquadTracker) -> str:
    """
    Returns a human-readable string of absent key players, e.g.
    'Halfback: D. Cherry-Evans, Hooker: L. Croker'
    Empty string if nobody absent.
    """
    if not players:
        return ""
    starter_map = {
        p["number"]: p
        for p in players
        if p["number"] in KEY_NUMBERS
    }
    NUM_TO_POS = {1: "Fullback", 6: "Five-Eighth", 7: "Halfback", 9: "Hooker"}
    absent = []
    for num in sorted(KEY_NUMBERS):
        regular_id = tracker.regular_starter(team, num)
        if regular_id is None:
            continue
        actual = starter_map.get(num)
        if actual is None or actual.get("playerId") != regular_id:
            pos = NUM_TO_POS.get(num, f"#{num}")
            if actual:
                name = f"{actual['firstName'][0]}. {actual['lastName']} (replacement)"
            else:
                name = "absent"
            absent.append(f"{pos}: {name}")
    return ", ".join(absent)
