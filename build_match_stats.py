"""Build the compact FIFA World Cup 2018 match-stat table from StatsBomb Open Data.

Run from any directory with:
    python build_match_stats.py

The generated CSV keeps the app lightweight; the original event feeds are not
downloaded again when Streamlit starts.
"""

import concurrent.futures
import csv
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MATCHES_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data/matches/43/3.json"
EVENT_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data/events/{match_id}.json"
OUTPUT_PATH = ROOT / "worldcup_match_stats.csv"
VALID_PERIODS = {1, 2, 3, 4}  # Exclude penalty shootouts from match statistics.
MAX_POSSESSION_GAP_SECONDS = 10.0


def fetch_json(url):
    """Fetch JSON with a small retry for transient GitHub errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "DUY-Football-Analytics"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def _seconds(timestamp):
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _blank_stats():
    return {
        "possession_seconds": 0.0,
        "shots": 0,
        "shots_on_target": 0,
        "shots_off_target": 0,
        "blocked_shots": 0,
        "xg": 0.0,
        "key_passes": 0,
        "goalkeeper_saves": 0,
        "corners": 0,
        "fouls": 0,
        "free_kicks": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "goals": 0,
    }


def _card_counts(stats, card_name):
    if card_name in {"Yellow Card", "Second Yellow"}:
        stats["yellow_cards"] += 1
    if card_name in {"Red Card", "Second Yellow"}:
        stats["red_cards"] += 1


def build_match_rows(match, events):
    home_team = match["home_team"]["home_team_name"]
    away_team = match["away_team"]["away_team_name"]
    team_names = {home_team, away_team}
    stats = {team: _blank_stats() for team in team_names}
    free_kick_pass_possessions = set()
    direct_free_kick_possessions = set()

    # Possession is estimated from short intervals between consecutive events
    # in the same possession. Long gaps are omitted because they can include
    # stoppages, and period 5 is the penalty shootout.
    for index, event in enumerate(events[:-1]):
        next_event = events[index + 1]
        period = event.get("period")
        team = (event.get("possession_team") or {}).get("name")
        if (
            period not in VALID_PERIODS
            or next_event.get("period") != period
            or event.get("possession") != next_event.get("possession")
            or team not in team_names
            or (next_event.get("possession_team") or {}).get("name") != team
        ):
            continue
        gap = _seconds(next_event["timestamp"]) - _seconds(event["timestamp"])
        if 0 < gap <= MAX_POSSESSION_GAP_SECONDS:
            stats[team]["possession_seconds"] += gap

    for event in events:
        if event.get("period") not in VALID_PERIODS:
            continue

        event_type = (event.get("type") or {}).get("name")
        team = (event.get("team") or {}).get("name")
        possession = event.get("possession")
        possession_key = (possession, team)

        if event_type == "Pass" and team in team_names:
            pass_data = event.get("pass") or {}
            pass_type = (pass_data.get("type") or {}).get("name")
            if pass_data.get("shot_assist"):
                stats[team]["key_passes"] += 1
            if pass_type == "Corner":
                stats[team]["corners"] += 1
            elif pass_type == "Free Kick":
                stats[team]["free_kicks"] += 1
                free_kick_pass_possessions.add(possession_key)

        elif event_type == "Shot" and team in team_names:
            shot = event.get("shot") or {}
            outcome = (shot.get("outcome") or {}).get("name")
            stats[team]["shots"] += 1
            stats[team]["xg"] += float(shot.get("statsbomb_xg") or 0.0)
            if outcome in {"Goal", "Saved", "Saved to Post"}:
                stats[team]["shots_on_target"] += 1
            elif outcome in {"Off T", "Wayward", "Post", "Saved Off T"}:
                stats[team]["shots_off_target"] += 1
            elif outcome == "Blocked":
                stats[team]["blocked_shots"] += 1
            if outcome == "Goal":
                stats[team]["goals"] += 1
            if outcome in {"Saved", "Saved to Post", "Saved Off T"}:
                defending_team = away_team if team == home_team else home_team
                stats[defending_team]["goalkeeper_saves"] += 1

            # Direct free-kick shots have no Free Kick pass event in their
            # possession, so count those once as a free-kick attempt.
            if (
                (event.get("play_pattern") or {}).get("name") == "From Free Kick"
                and possession_key not in free_kick_pass_possessions
            ):
                direct_free_kick_possessions.add(possession_key)

        elif event_type == "Foul Committed" and team in team_names:
            stats[team]["fouls"] += 1
            card = ((event.get("foul_committed") or {}).get("card") or {}).get("name")
            _card_counts(stats[team], card)

        elif event_type == "Bad Behaviour" and team in team_names:
            card = ((event.get("bad_behaviour") or {}).get("card") or {}).get("name")
            _card_counts(stats[team], card)

    for possession_key in direct_free_kick_possessions:
        team = possession_key[1]
        if team in team_names:
            stats[team]["free_kicks"] += 1

    total_possession_seconds = sum(item["possession_seconds"] for item in stats.values())
    rows = []
    for team, side in ((home_team, "Home"), (away_team, "Away")):
        values = stats[team]
        possession_pct = (
            values["possession_seconds"] / total_possession_seconds * 100
            if total_possession_seconds > 0
            else ""
        )
        rows.append({
            "match_id": match["match_id"],
            "match_date": match.get("match_date", ""),
            "competition_stage": (match.get("competition_stage") or {}).get("name", ""),
            "home_team": home_team,
            "away_team": away_team,
            "team_name": team,
            "team_side": side,
            "possession_pct": round(possession_pct, 1) if possession_pct != "" else "",
            **values,
            "source": "StatsBomb Open Data — FIFA World Cup 2018",
        })
    return rows


def main():
    matches = fetch_json(MATCHES_URL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        event_feeds = list(pool.map(
            lambda match: (match, fetch_json(EVENT_URL.format(match_id=match["match_id"]))),
            matches,
        ))

    rows = []
    for match, events in event_feeds:
        rows.extend(build_match_rows(match, events))

    fieldnames = list(rows[0])
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} team rows for {len(matches)} matches to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
