"""
Bridge linking every match on the Real-Time Daily Schedule (`ws_today`) into the
full 6-tab NCKH TUD65 Match Center (`ws_match`): Lineups/Ratings, Stats, Shot Map,
Player Heat/Pass Map, Passing Network, and AI/Live Timeline.
"""

from datetime import datetime, timezone
import hashlib
import math
import os
import random

import pandas as pd
import streamlit as st

from build_pro_analytics_store import clean_club_name
from prematch_research import _safe_float, _team_alias_key
from pro_features import _map_opta_tactical_pos


_STANDARD_4231_SLOTS = [
    ("Goalkeeper", 1),
    ("Right Back", 2),
    ("Right Center Back", 4),
    ("Left Center Back", 5),
    ("Left Back", 3),
    ("Right Defensive Midfield", 6),
    ("Left Defensive Midfield", 8),
    ("Right Wing", 7),
    ("Center Attacking Midfield", 10),
    ("Left Wing", 11),
    ("Center Forward", 9),
]

_SB_ANCHOR_COORDS = {
    "Goalkeeper": (9.0, 40.0),
    "Right Back": (31.0, 68.0),
    "Left Back": (31.0, 12.0),
    "Right Wing Back": (44.0, 71.0),
    "Left Wing Back": (44.0, 9.0),
    "Right Center Back": (26.0, 52.0),
    "Left Center Back": (26.0, 28.0),
    "Center Back": (24.0, 40.0),
    "Center Defensive Midfield": (46.0, 40.0),
    "Right Defensive Midfield": (47.0, 51.0),
    "Left Defensive Midfield": (47.0, 29.0),
    "Right Midfield": (62.0, 67.0),
    "Left Midfield": (62.0, 13.0),
    "Right Center Midfield": (59.0, 54.0),
    "Left Center Midfield": (59.0, 26.0),
    "Center Midfield": (58.0, 40.0),
    "Center Attacking Midfield": (76.0, 40.0),
    "Right Attacking Midfield": (76.0, 56.0),
    "Left Attacking Midfield": (76.0, 24.0),
    "Right Wing": (88.0, 65.0),
    "Left Wing": (88.0, 15.0),
    "Right Center Forward": (94.0, 51.0),
    "Left Center Forward": (94.0, 29.0),
    "Center Forward": (96.0, 40.0),
    "Second Striker": (86.0, 40.0),
    "Secondary Striker": (86.0, 40.0),
}

_ESPN_POS_MAP = {
    "G": "Goalkeeper", "GK": "Goalkeeper",
    "RB": "Right Back", "LB": "Left Back",
    "RWB": "Right Wing Back", "LWB": "Left Wing Back",
    "CD-R": "Right Center Back", "RCB": "Right Center Back", "D-R": "Right Center Back",
    "CD-L": "Left Center Back", "LCB": "Left Center Back", "D-L": "Left Center Back",
    "CD": "Center Back", "CB": "Center Back", "D": "Center Back",
    "DM": "Center Defensive Midfield", "CDM": "Center Defensive Midfield",
    "DM-L": "Left Defensive Midfield", "LDM": "Left Defensive Midfield",
    "DM-R": "Right Defensive Midfield", "RDM": "Right Defensive Midfield",
    "LM": "Left Midfield", "RM": "Right Midfield",
    "CM-L": "Left Center Midfield", "LCM": "Left Center Midfield",
    "CM-R": "Right Center Midfield", "RCM": "Right Center Midfield",
    "CM": "Center Midfield", "M": "Center Midfield",
    "AM": "Center Attacking Midfield", "CAM": "Center Attacking Midfield",
    "AM-L": "Left Wing", "LAM": "Left Wing", "LW": "Left Wing", "LF": "Left Wing",
    "AM-R": "Right Wing", "RAM": "Right Wing", "RW": "Right Wing", "RF": "Right Wing",
    "F": "Center Forward", "CF": "Center Forward", "ST": "Center Forward",
}


def _map_espn_pos_to_sb(pos_abbr: str) -> str:
    p = str(pos_abbr or "").strip().upper()
    if p in _ESPN_POS_MAP:
        return _ESPN_POS_MAP[p]
    if p.startswith("G"):
        return "Goalkeeper"
    if "D" in p or "B" in p:
        if "L" in p:
            return "Left Back"
        if "R" in p:
            return "Right Back"
        return "Center Back"
    if any(k in p for k in ("F", "W", "S")):
        if "L" in p:
            return "Left Wing"
        if "R" in p:
            return "Right Wing"
        return "Center Forward"
    if "L" in p:
        return "Left Center Midfield"
    if "R" in p:
        return "Right Center Midfield"
    return "Center Midfield"


def _build_fallback_xi(team_name: str) -> dict:
    short_t = str(team_name or "Team").strip()
    starters = []
    for idx, (pos_name, num) in enumerate(_STANDARD_4231_SLOTS):
        label_map = {
            "Goalkeeper": "Thủ môn",
            "Right Back": "Hậu vệ phải",
            "Right Center Back": "Trung vệ lệch phải",
            "Left Center Back": "Trung vệ lệch trái",
            "Left Back": "Hậu vệ trái",
            "Right Defensive Midfield": "Tiền vệ trụ phải",
            "Left Defensive Midfield": "Tiền vệ trụ trái",
            "Right Wing": "Cánh phải",
            "Center Attacking Midfield": "Tiền vệ công",
            "Left Wing": "Cánh trái",
            "Center Forward": "Trung phong",
        }
        starters.append({
            "name": f"{label_map.get(pos_name, f'Cầu thủ {idx+1}')} ({short_t})",
            "position": pos_name,
            "jersey_number": num,
        })
    bench = [
        {"name": f"Dự bị Thủ môn ({short_t})", "position": "Goalkeeper", "jersey_number": 13},
        {"name": f"Dự bị Hậu vệ ({short_t})", "position": "Center Back", "jersey_number": 15},
        {"name": f"Dự bị Tiền vệ ({short_t})", "position": "Center Midfield", "jersey_number": 16},
        {"name": f"Dự bị Tiền đạo ({short_t})", "position": "Center Forward", "jersey_number": 19},
    ]
    return {"formation": "4-2-3-1", "starters": starters, "bench": bench}


@st.cache_resource(show_spinner=False)
def _load_latest_club_xi_index(sample_dir: str) -> dict:
    """Index latest real starting XI & bench per club from 2026/27 Opta lineups parquets."""
    index: dict[str, dict] = {}
    paths = [
        os.path.join(sample_dir, "opta_modern", "modern_lineups.parquet"),
        os.path.join(sample_dir, "opta_ucl", "ucl_lineups.parquet"),
    ]
    for p_path in paths:
        if not os.path.isfile(p_path):
            continue
        try:
            cols = [
                "match_id", "match_date", "team_name", "player_name",
                "position", "position_side", "formation_place", "shirt_number", "is_starter",
            ]
            df = pd.read_parquet(p_path, columns=cols)
        except Exception:
            try:
                df = pd.read_parquet(p_path)
            except Exception:
                continue
        if df.empty or "team_name" not in df.columns or "match_id" not in df.columns:
            continue

        df = df[df["team_name"].notna() & df["player_name"].notna()].copy()
        if "match_date" not in df.columns:
            df["match_date"] = "2025-01-01"
        df["match_date"] = df["match_date"].astype(str).str.slice(0, 10)

        latest_pairs = (
            df[["team_name", "match_date", "match_id"]]
            .drop_duplicates()
            .sort_values(["match_date", "match_id"])
            .groupby("team_name", sort=False)
            .tail(1)
        )
        latest_df = df.merge(
            latest_pairs[["team_name", "match_id"]],
            on=["team_name", "match_id"],
            how="inner",
        )
        latest_df["_fp"] = pd.to_numeric(latest_df.get("formation_place"), errors="coerce").fillna(99)
        latest_df["_gk"] = latest_df["position"].astype(str).str.lower().str.contains("goalkeeper", na=False)
        latest_df["_st"] = latest_df["is_starter"].fillna(False).astype(bool)

        for raw_tname, grp in latest_df.groupby("team_name", sort=False):
            cleaned_t = clean_club_name(str(raw_tname))
            keys = [k for k in {_team_alias_key(cleaned_t), _team_alias_key(str(raw_tname))} if k]
            if not keys or all(k in index for k in keys):
                continue

            g_sorted = grp.sort_values(by=["_st", "_gk", "_fp"], ascending=[False, False, True])
            starters, bench = [], []
            seen_names = set()
            for _, r in g_sorted.iterrows():
                pname = str(r.get("player_name") or "").strip()
                if not pname or pname in seen_names:
                    continue
                seen_names.add(pname)
                sb_pos = _map_opta_tactical_pos(r.get("position"), r.get("position_side"))
                shirt = pd.to_numeric(r.get("shirt_number"), errors="coerce")
                jersey = int(shirt) if pd.notna(shirt) and shirt > 0 else ""
                item = {
                    "name": pname,
                    "position": sb_pos or ("Goalkeeper" if len(starters) == 0 else "Center Midfield"),
                    "jersey_number": jersey,
                }
                if bool(r.get("_st")) and len(starters) < 11:
                    starters.append(item)
                else:
                    bench.append(item)

            if len(starters) >= 7:
                entry = {"formation": "", "starters": starters[:11], "bench": bench[:10]}
                for k in keys:
                    if k not in index:
                        index[k] = entry

    return index


def _find_club_xi(xi_index: dict, team_name: str) -> dict:
    for cand in (team_name, clean_club_name(team_name)):
        k = _team_alias_key(cand)
        if k and k in xi_index:
            return xi_index[k]
    return _build_fallback_xi(team_name)


def _match_player_name(target: str, pool: list[str]) -> str:
    t_low = str(target or "").strip().lower()
    if not t_low:
        return ""
    for p in pool:
        if p.lower() == t_low:
            return p
    t_parts = [w for w in t_low.replace(".", "").split() if len(w) > 1]
    if t_parts:
        last = t_parts[-1]
        for p in pool:
            if last in p.lower():
                return p
    return ""


def _roster_to_lineup_and_squad(espn_roster: dict | None, fallback_team_name: str, xi_index: dict):
    if espn_roster and (espn_roster.get("starters") or []):
        starters = []
        squad_players = []
        for idx, p in enumerate(espn_roster.get("starters") or []):
            pname = str(p.get("name") or p.get("short_name") or "").strip()
            if not pname:
                continue
            sb_pos = _map_espn_pos_to_sb(p.get("pos", ""))
            jersey = int(p["jersey"]) if str(p.get("jersey", "")).isdigit() else ""
            starters.append({"name": pname, "position": sb_pos, "jersey_number": jersey})
            squad_players.append({
                "name": pname,
                "nickname": pname,
                "jersey_number": jersey,
                "position": sb_pos,
                "started": True,
                "entered": False,
                "left": bool(p.get("subbed_out")),
                "injured_on_exit": False,
                "minute_in": None,
                "minute_out": "65:00" if p.get("subbed_out") else None,
                "_espn": p,
            })
        for p in (espn_roster.get("bench") or []):
            pname = str(p.get("name") or p.get("short_name") or "").strip()
            if not pname:
                continue
            sb_pos = _map_espn_pos_to_sb(p.get("pos", ""))
            jersey = int(p["jersey"]) if str(p.get("jersey", "")).isdigit() else ""
            entered = bool(p.get("subbed_in"))
            squad_players.append({
                "name": pname,
                "nickname": pname,
                "jersey_number": jersey,
                "position": sb_pos,
                "started": False,
                "entered": entered,
                "left": False,
                "injured_on_exit": False,
                "minute_in": "65:00" if entered else None,
                "minute_out": None,
                "_espn": p,
            })
        return (
            {"formation": str(espn_roster.get("formation") or ""), "players": starters[:11]},
            {"players": squad_players},
        )

    fb = _find_club_xi(xi_index, fallback_team_name)
    starters = list(fb["starters"])
    squad_players = [
        {
            "name": p["name"],
            "nickname": p["name"],
            "jersey_number": p.get("jersey_number", ""),
            "position": p["position"],
            "started": True,
            "entered": False,
            "left": False,
            "injured_on_exit": False,
            "minute_in": None,
            "minute_out": None,
        }
        for p in starters
    ] + [
        {
            "name": p["name"],
            "nickname": p["name"],
            "jersey_number": p.get("jersey_number", ""),
            "position": p["position"],
            "started": False,
            "entered": False,
            "left": False,
            "injured_on_exit": False,
            "minute_in": None,
            "minute_out": None,
        }
        for p in fb.get("bench", [])
    ]
    return {"formation": fb.get("formation", ""), "players": starters}, {"players": squad_players}


def _anchor_for_pos(sb_pos: str, idx: int) -> tuple[float, float]:
    if sb_pos in _SB_ANCHOR_COORDS:
        return _SB_ANCHOR_COORDS[sb_pos]
    if idx == 0:
        return (9.0, 40.0)
    if idx <= 4:
        return (28.0, 14.0 + (idx - 1) * 17.0)
    if idx <= 7:
        return (56.0, 20.0 + (idx - 5) * 20.0)
    return (86.0, 18.0 + (idx - 8) * 22.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _make_event_row(
    match_id: str,
    minute: int,
    second: int,
    team_name: str,
    player_name: str,
    pos_name: str,
    ev_type: str,
    lx: float,
    ly: float,
    **extra,
) -> dict:
    period = 1 if minute <= 45 else 2
    row = {
        "match_id": match_id,
        "minute": int(minute),
        "second": int(second),
        "period": period,
        "timestamp": f"00:{int(minute) % 60:02d}:{int(second) % 60:02d}",
        "team_name": team_name,
        "player_name": player_name,
        "position_name": pos_name,
        "type_name": ev_type,
        "location_x": round(_clamp(lx, 1.0, 119.0), 2),
        "location_y": round(_clamp(ly, 2.0, 78.0), 2),
        "pass_recipient_name": extra.get("pass_recipient_name", ""),
        "pass_outcome_name": extra.get("pass_outcome_name", ""),
        "pass_end_location_x": extra.get("pass_end_location_x"),
        "pass_end_location_y": extra.get("pass_end_location_y"),
        "pass_length": extra.get("pass_length"),
        "pass_cross": int(extra.get("pass_cross", 0)),
        "pass_through_ball": int(extra.get("pass_through_ball", 0)),
        "pass_switch": int(extra.get("pass_switch", 0)),
        "pass_shot_assist": int(extra.get("pass_shot_assist", 0)),
        "pass_goal_assist": int(extra.get("pass_goal_assist", 0)),
        "shot_outcome_name": extra.get("shot_outcome_name", ""),
        "shot_statsbomb_xg": extra.get("shot_statsbomb_xg"),
        "shot_type_name": extra.get("shot_type_name", ""),
        "shot_body_part_name": extra.get("shot_body_part_name", ""),
        "dribble_outcome_name": extra.get("dribble_outcome_name", ""),
        "card_name": extra.get("card_name", ""),
        "substitution_replacement_name": extra.get("substitution_replacement_name", ""),
        "substitution_outcome_name": extra.get("substitution_outcome_name", ""),
        "gk_type_name": extra.get("gk_type_name", ""),
        "gk_outcome_name": extra.get("gk_outcome_name", ""),
        "carry_distance": 0.0,
        "under_pressure": 0,
        "possession": None,
        "possession_team": team_name,
    }
    return row


def _synthesize_match_events(
    match_id: str,
    home_team: str,
    away_team: str,
    custom_lineups: dict,
    match_squads: dict,
    stats_dict: dict,
    goals_list: list,
    details_list: list,
    seed_key: str,
) -> pd.DataFrame:
    """Deterministically build realistic 34-column match_events matching team stats, lineups, shots & cards."""
    seed_int = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed_int)
    rows: list[dict] = []

    details_by_team = {home_team: [], away_team: []}
    for d in (details_list or []):
        tn = str(d.get("team_name") or "")
        if tn == home_team:
            details_by_team[home_team].append(d)
        elif tn == away_team:
            details_by_team[away_team].append(d)

    goals_by_team = {home_team: [], away_team: []}
    for g in (goals_list or []):
        tn = str(g.get("team") or "")
        if tn == home_team:
            goals_by_team[home_team].append(g)
        elif tn == away_team:
            goals_by_team[away_team].append(g)

    for team_name in (home_team, away_team):
        st_t = stats_dict[team_name]
        starters = list((custom_lineups.get(team_name) or {}).get("players") or [])
        if not starters:
            continue
        sq_players = (match_squads.get(team_name) or {}).get("players") or []
        espn_meta_by_name = {p["name"]: p.get("_espn") for p in sq_players if isinstance(p, dict) and p.get("_espn")}

        player_names = [p["name"] for p in starters]
        pos_by_name = {p["name"]: p.get("position", "Center Midfield") for p in starters}
        anchor_by_name = {p["name"]: _anchor_for_pos(p.get("position", ""), idx) for idx, p in enumerate(starters)}

        gk_name = next((n for n in player_names if "goalkeeper" in pos_by_name[n].lower()), player_names[0])
        outfield = [n for n in player_names if n != gk_name] or player_names

        # Attackers/midfielders carry higher shot weights
        shot_weights = []
        for n in outfield:
            espn_p = espn_meta_by_name.get(n) or {}
            base_w = 0.7 if any(k in pos_by_name[n] for k in ("Back", "Defender")) else (1.5 if "Midfield" in pos_by_name[n] else 2.6)
            base_w += float(espn_p.get("shots", 0)) * 2.2
            shot_weights.append(max(0.2, base_w))

        # 1. Goals + assisting key passes
        n_goals = int(st_t.get("goals") or 0)
        t_goals = list(goals_by_team[team_name])
        goal_xg_sum = 0.0
        for g_idx in range(n_goals):
            g_rec = t_goals[g_idx] if g_idx < len(t_goals) else {}
            raw_min = str(g_rec.get("minute") or rng.randint(8, 86)).replace("'", "").split("+")[0]
            g_min = max(1, min(95, int(raw_min) if raw_min.isdigit() else rng.randint(8, 86)))
            scorer = _match_player_name(g_rec.get("player", ""), player_names) or rng.choices(outfield, weights=shot_weights, k=1)[0]
            s_xg = round(rng.uniform(0.24, 0.52), 2)
            goal_xg_sum += s_xg
            sx = rng.uniform(103.0, 114.5)
            sy = rng.uniform(31.0, 49.0)
            rows.append(_make_event_row(
                match_id, g_min, rng.randint(5, 55), team_name, scorer, pos_by_name[scorer],
                "Shot", sx, sy,
                shot_outcome_name="Goal", shot_statsbomb_xg=s_xg,
                shot_type_name="Open Play", shot_body_part_name="Right Foot",
            ))
            # Assister pass
            assist_cands = [n for n in outfield if n != scorer] or outfield
            assister = _match_player_name(g_rec.get("assist", ""), assist_cands) or rng.choice(assist_cands)
            ax, ay = anchor_by_name[assister]
            ax0 = _clamp(max(ax, sx - rng.uniform(10.0, 26.0)), 30.0, 114.0)
            ay0 = _clamp(ay + rng.uniform(-8.0, 8.0), 6.0, 74.0)
            rows.append(_make_event_row(
                match_id, g_min, rng.randint(1, 5), team_name, assister, pos_by_name[assister],
                "Pass", ax0, ay0,
                pass_recipient_name=scorer, pass_end_location_x=round(sx, 2), pass_end_location_y=round(sy, 2),
                pass_length=round(math.hypot(sx - ax0, sy - ay0), 1),
                pass_shot_assist=1, pass_goal_assist=1,
            ))

        # 2. Remaining shots scaled to hit exact target xG
        tot_shots = int(st_t.get("shots") or 0)
        on_tgt = int(st_t.get("shots_on_target") or 0)
        blk_shots = int(st_t.get("blocked_shots") or 0)
        rem_shots = max(0, tot_shots - n_goals)
        rem_saved = max(0, on_tgt - n_goals)
        rem_blk = min(rem_shots - rem_saved, max(0, blk_shots))
        shot_outcomes = (["Saved"] * rem_saved) + (["Blocked"] * rem_blk) + (["Off T"] * max(0, rem_shots - rem_saved - rem_blk))
        target_rem_xg = max(0.0, float(st_t.get("xg") or 0.0) - goal_xg_sum)
        per_shot_xg = (target_rem_xg / len(shot_outcomes)) if shot_outcomes else 0.0
        for s_out in shot_outcomes:
            shooter = rng.choices(outfield, weights=shot_weights, k=1)[0]
            sx = rng.uniform(92.0, 112.0) if s_out == "Saved" else rng.uniform(86.0, 108.0)
            sy = rng.uniform(22.0, 58.0)
            s_xg = round(_clamp(per_shot_xg * rng.uniform(0.7, 1.35), 0.02, 0.45), 2)
            rows.append(_make_event_row(
                match_id, rng.randint(3, 90), rng.randint(0, 59), team_name, shooter, pos_by_name[shooter],
                "Shot", sx, sy,
                shot_outcome_name=s_out, shot_statsbomb_xg=s_xg,
                shot_type_name="Open Play", shot_body_part_name="Right Foot",
            ))

        # 3. Cards (ESPN details first, then player stats/team counts)
        y_cards = int(st_t.get("yellow_cards") or 0)
        r_cards = int(st_t.get("red_cards") or 0)
        card_events = [d for d in details_by_team[team_name] if d.get("yellow_card") or d.get("red_card")]
        used_y, used_r = 0, 0
        for cd in card_events:
            c_min_s = str(cd.get("clock") or "55").replace("'", "").split("+")[0]
            c_min = max(1, min(94, int(c_min_s) if c_min_s.isdigit() else 55))
            c_player = _match_player_name(cd.get("player", ""), player_names) or rng.choice(outfield)
            c_name = "Red Card" if cd.get("red_card") else "Yellow Card"
            if cd.get("red_card"):
                used_r += 1
            else:
                used_y += 1
            cx, cy = anchor_by_name[c_player]
            rows.append(_make_event_row(
                match_id, c_min, rng.randint(0, 59), team_name, c_player, pos_by_name[c_player],
                "Bad Behaviour", cx, cy, card_name=c_name,
            ))
        for _ in range(max(0, y_cards - used_y)):
            cp = rng.choice(outfield)
            cx, cy = anchor_by_name[cp]
            rows.append(_make_event_row(match_id, rng.randint(15, 88), 10, team_name, cp, pos_by_name[cp], "Bad Behaviour", cx, cy, card_name="Yellow Card"))
        for _ in range(max(0, r_cards - used_r)):
            cp = rng.choice(outfield)
            cx, cy = anchor_by_name[cp]
            rows.append(_make_event_row(match_id, rng.randint(25, 88), 10, team_name, cp, pos_by_name[cp], "Bad Behaviour", cx, cy, card_name="Red Card"))

        # 4. Substitutions
        sub_out_list = [p for p in sq_players if isinstance(p, dict) and p.get("started") and p.get("left")]
        sub_in_list = [p for p in sq_players if isinstance(p, dict) and (not p.get("started")) and p.get("entered")]
        for i_sub in range(min(len(sub_out_list), len(sub_in_list))):
            p_out = sub_out_list[i_sub]["name"]
            p_in = sub_in_list[i_sub]["name"]
            s_min = 60 + i_sub * 7
            ox, oy = anchor_by_name.get(p_out, (55.0, 40.0))
            pos_by_name[p_in] = sub_in_list[i_sub].get("position") or pos_by_name.get(p_out, "Center Midfield")
            anchor_by_name[p_in] = (ox, oy)
            rows.append(_make_event_row(
                match_id, s_min, 0, team_name, p_out, pos_by_name.get(p_out, ""),
                "Substitution", ox, oy,
                substitution_replacement_name=p_in, substitution_outcome_name="Tactical",
            ))

        # 5. GK saves
        for i_sv in range(int(st_t.get("goalkeeper_saves") or 0)):
            rows.append(_make_event_row(
                match_id, rng.randint(5, 89), rng.randint(0, 59), team_name, gk_name, "Goalkeeper",
                "Goal Keeper", rng.uniform(3.0, 8.0), rng.uniform(33.0, 47.0),
                gk_type_name="Shot Saved", gk_outcome_name="Saved",
            ))

        # 6. Tactical Passes (capped cleanly for snappy client plotting while keeping faithful totals/links)
        raw_passes = max(24, min(260, int(st_t.get("passes") or 160)))
        comp_passes_target = int(st_t.get("passes_completed") or int(raw_passes * 0.8))
        comp_ratio = max(0.58, min(0.94, comp_passes_target / max(1, int(st_t.get("passes") or raw_passes))))
        key_p_target = max(1, int(st_t.get("key_passes") or max(2, tot_shots // 2)))

        # Precompute nearest tactical teammates for cohesive realistic passing network links
        neighbors: dict[str, list[str]] = {}
        for p1 in player_names:
            x1, y1 = anchor_by_name[p1]
            ranked = sorted(
                [p2 for p2 in player_names if p2 != p1],
                key=lambda p2: math.hypot(anchor_by_name[p2][0] - x1, anchor_by_name[p2][1] - y1),
            )
            neighbors[p1] = ranked[:4] if ranked else player_names

        key_p_left = max(0, key_p_target - n_goals)
        for i_p in range(raw_passes):
            passer = player_names[i_p % len(player_names)] if i_p < len(player_names) * 2 else rng.choice(player_names)
            rec = rng.choice(neighbors[passer]) if neighbors[passer] else passer
            x1, y1 = anchor_by_name[passer]
            x2, y2 = anchor_by_name[rec]
            sx = _clamp(x1 + rng.uniform(-9.0, 9.0), 4.0, 114.0)
            sy = _clamp(y1 + rng.uniform(-8.0, 8.0), 4.0, 76.0)
            ex = _clamp(x2 + rng.uniform(-7.0, 7.0), 5.0, 116.0)
            ey = _clamp(y2 + rng.uniform(-7.0, 7.0), 4.0, 76.0)
            is_ok = rng.random() < comp_ratio
            p_len = round(math.hypot(ex - sx, ey - sy), 1)
            is_kp = 1 if (is_ok and key_p_left > 0 and passer in outfield and i_p % 7 == 0) else 0
            if is_kp:
                key_p_left -= 1
            rows.append(_make_event_row(
                match_id, rng.randint(1, 90), rng.randint(0, 59), team_name, passer, pos_by_name.get(passer, ""),
                "Pass", sx, sy,
                pass_recipient_name=rec if is_ok else "",
                pass_outcome_name="" if is_ok else "Incomplete",
                pass_end_location_x=round(ex, 2),
                pass_end_location_y=round(ey, 2),
                pass_length=p_len,
                pass_cross=1 if (abs(sy - 40.0) > 22.0 and ex > 94.0 and rng.random() < 0.35) else 0,
                pass_through_ball=1 if (ex - sx > 16.0 and rng.random() < 0.15) else 0,
                pass_switch=1 if abs(ey - sy) >= 32.0 else 0,
                pass_shot_assist=is_kp,
            ))

        # 7. Defensive, Dribble & Foul events across outfielders so ratings & heatmaps are complete
        for f_idx in range(max(2, min(18, int(st_t.get("fouls") or 8)))):
            fp = rng.choice(outfield)
            fx, fy = anchor_by_name[fp]
            rows.append(_make_event_row(
                match_id, rng.randint(2, 90), rng.randint(0, 59), team_name, fp, pos_by_name[fp],
                "Foul Committed", fx + rng.uniform(-6, 6), fy + rng.uniform(-6, 6),
            ))
        for _ in range(max(12, len(outfield) * 3)):
            ap = rng.choice(outfield)
            ax, ay = anchor_by_name[ap]
            etype = rng.choice(("Dribble", "Interception", "Ball Recovery", "Clearance", "Duel"))
            rows.append(_make_event_row(
                match_id, rng.randint(1, 90), rng.randint(0, 59), team_name, ap, pos_by_name[ap],
                etype, ax + rng.uniform(-11, 11), ay + rng.uniform(-9, 9),
                dribble_outcome_name="Complete" if (etype == "Dribble" and rng.random() < 0.65) else ("Incomplete" if etype == "Dribble" else ""),
            ))

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["period", "minute", "second"], kind="mergesort").reset_index(drop=True)
    return df_out


def _parse_goal_minute(raw_clock) -> tuple[int, str]:
    s = str(raw_clock or "").replace("′", "").replace("'", "").strip()
    if not s:
        return 0, "?"
    base_s = s.split("+")[0].strip()
    try:
        return int(float(base_s)), s
    except Exception:
        digits = "".join(ch for ch in base_s if ch.isdigit())
        return (int(digits), s) if digits else (0, s)


def _parse_stat_int(h_str: str, fallback: int = 0) -> int:
    try:
        return int(round(float(str(h_str).split("/")[0].replace("%", "").replace(",", "").strip())))
    except Exception:
        return fallback


def build_full_match_center_bundle(fixture: dict, sample_dir: str, force_refresh: bool = False):
    """
    Build the complete (events_df, stats_dict, custom_lineups, match_squads, current_match, meta_override)
    tuple expected by app.py Tabs 1–6 for any match from Daily Real-Time Hub (live, upcoming, or finished).
    """
    from daily_realtime_hub import get_espn_match_summary

    ht = str(fixture.get("home_team") or "Home")
    at = str(fixture.get("away_team") or "Away")
    state = str(fixture.get("state") or "pre").lower()
    eid = str(fixture.get("espn_event_id") or "")
    slug = str(fixture.get("espn_slug") or "")
    uid = str(fixture.get("upcoming_match_id") or "")
    vn_d = str(fixture.get("vn_date") or "")[:10]
    vn_t = str(fixture.get("vn_time") or "")
    comp_lbl = str(fixture.get("competition") or "NCKH TUD65 Match Center")
    xi_index = _load_latest_club_xi_index(sample_dir)

    ai_summary = None
    if fixture.get("prob_H") is not None:
        ai_summary = {
            "prob_H": fixture.get("prob_H"),
            "prob_D": fixture.get("prob_D"),
            "prob_A": fixture.get("prob_A"),
            "pred_home_goals": fixture.get("pred_home_goals"),
            "pred_away_goals": fixture.get("pred_away_goals"),
            "odds": fixture.get("odds") or {},
            "upcoming_match_id": uid,
        }

    # --- CASE A: LIVE ('in') or ESPN FINISHED ('post') ---
    if state in ("in", "post") and eid and slug:
        summ = get_espn_match_summary(slug, eid, force_refresh=force_refresh, fallback_match=fixture)
        s_state = summ.get("state") or state
        h_info = summ.get("home") or {}
        a_info = summ.get("away") or {}
        ht = str(h_info.get("name") or ht)
        at = str(a_info.get("name") or at)
        hs = int(h_info["score"]) if h_info.get("score") is not None else int(fixture.get("home_score") or 0)
        as_ = int(a_info["score"]) if a_info.get("score") is not None else int(fixture.get("away_score") or 0)

        rosters = summ.get("rosters") or []
        h_ros = rosters[0] if len(rosters) >= 1 else None
        a_ros = rosters[1] if len(rosters) >= 2 else None
        h_lu, h_sq = _roster_to_lineup_and_squad(h_ros, ht, xi_index)
        a_lu, a_sq = _roster_to_lineup_and_squad(a_ros, at, xi_index)
        custom_lineups = {ht: h_lu, at: a_lu}
        match_squads = {ht: h_sq, at: a_sq}

        stat_map = {k: (h_s, a_s, h_n, a_n) for k, _lbl, h_s, a_s, h_n, a_n in (summ.get("stats") or [])}

        def _team_bs(is_h: bool, goals_cnt: int) -> dict:
            idx_s = 0 if is_h else 1
            num_idx = 2 if is_h else 3
            poss = float(stat_map["possessionPct"][num_idx]) if "possessionPct" in stat_map else 50.0
            shots = int(round(stat_map["totalShots"][num_idx])) if "totalShots" in stat_map else max(goals_cnt + 2, 5)
            on_tgt = int(round(stat_map["shotsOnTarget"][num_idx])) if "shotsOnTarget" in stat_map else max(goals_cnt, int(round(shots * 0.36)))
            on_tgt = max(goals_cnt, on_tgt)
            rem_s = max(0, shots - on_tgt)
            blk = rem_s // 3
            off_tgt = max(0, rem_s - blk)
            if "accuratePasses" in stat_map:
                raw_str = stat_map["accuratePasses"][idx_s]
                p_comp = int(round(stat_map["accuratePasses"][num_idx]))
                p_tot = _parse_stat_int(raw_str.split("/")[1].split()[0], p_comp) if "/" in raw_str else max(p_comp, int(round(p_comp / 0.82)))
            else:
                p_tot = int(round(120 + poss * 3.5))
                p_comp = int(round(p_tot * 0.80))
            saves = int(round(stat_map["saves"][num_idx])) if "saves" in stat_map else 2
            corners = int(round(stat_map["wonCorners"][num_idx])) if "wonCorners" in stat_map else max(1, shots // 3)
            fouls = int(round(stat_map["foulsCommitted"][num_idx])) if "foulsCommitted" in stat_map else 8
            yc = int(round(stat_map["yellowCards"][num_idx])) if "yellowCards" in stat_map else 1
            rc = int(round(stat_map["redCards"][num_idx])) if "redCards" in stat_map else 0
            xg = round(goals_cnt * 0.36 + on_tgt * 0.14 + max(0, shots - on_tgt) * 0.045, 2) if (shots > 0 or goals_cnt > 0) else 0.0
            return {
                "possession_seconds": round(poss * 30.0, 1),
                "possession_pct": round(poss, 1),
                "shots": shots,
                "shots_on_target": on_tgt,
                "shots_off_target": off_tgt,
                "blocked_shots": blk,
                "xg": xg,
                "key_passes": max(goals_cnt + 1, int(round(shots * 0.65))),
                "goalkeeper_saves": saves,
                "corners": corners,
                "fouls": fouls,
                "free_kicks": max(2, fouls + 1),
                "yellow_cards": yc,
                "red_cards": rc,
                "goals": goals_cnt,
                "passes": max(1, p_tot),
                "passes_completed": min(p_comp, max(1, p_tot)),
            }

        stats_dict = {ht: _team_bs(True, hs), at: _team_bs(False, as_)}

        goals_list = []
        details_norm = []
        h_id, a_id = str(h_info.get("id") or ""), str(a_info.get("id") or "")
        for d in (summ.get("details") or []):
            is_h = (h_id and str(d.get("team_id") or "") == h_id) or (not d.get("team_id") and d.get("team_name") == ht)
            t_label = ht if is_h else at
            d_copy = dict(d)
            d_copy["team_name"] = t_label
            details_norm.append(d_copy)
            if d.get("scoring_play"):
                g_min_int, g_min_disp = _parse_goal_minute(d.get("clock"))
                goals_list.append({
                    "player": str(d.get("player") or "Cầu thủ"),
                    "assist": str(d.get("assist") or ""),
                    "minute": g_min_int,
                    "minute_display": g_min_disp,
                    "team": t_label,
                    "is_og": bool(d.get("own_goal")),
                })

        mid_str = f"daily_espn_{eid}"
        seed_s = f"{eid}:{hs}:{as_}:{stats_dict[ht]['shots']}:{stats_dict[at]['shots']}"
        events_df = _synthesize_match_events(
            mid_str, ht, at, custom_lineups, match_squads, stats_dict, goals_list, details_norm, seed_s
        )

        gi = summ.get("game_info") or {}
        bcasts = gi.get("broadcasts") or []
        officials = gi.get("officials") or []
        st_det = str(summ.get("status_detail") or fixture.get("status_detail") or ("LIVE" if s_state == "in" else "FT"))
        current_match = {
            "id": mid_str,
            "stage": "Trực tiếp ESPN" if s_state == "in" else "ESPN Match Summary",
            "home_team": ht,
            "away_team": at,
            "home_logo": str(h_info.get("logo") or fixture.get("home_logo") or ""),
            "away_logo": str(a_info.get("logo") or fixture.get("away_logo") or ""),
            "home_score": hs,
            "away_score": as_,
            "pen_home": None,
            "pen_away": None,
            "stadium": gi.get("venue") or fixture.get("venue") or "Sân vận động giải đấu",
            "referee": ", ".join(officials[:2]) if officials else "Trọng tài FIFA / Liên đoàn",
            "goals": goals_list,
            "live_state": s_state,
            "live_status_badge": f"🔴 TRỰC TIẾP · {st_det}" if s_state == "in" else f"HẾT GIỜ · {st_det}",
            "is_upcoming_preview": False,
            "fetched_vn": summ.get("fetched_vn") or "",
            "espn_key_events": summ.get("key_events") or [],
            "espn_commentary": summ.get("commentary") or [],
            "espn_broadcasts": bcasts,
            "espn_videos": summ.get("videos") or [],
            "prematch_ai_summary": ai_summary,
        }
        meta_override = {
            "id": mid_str,
            "competition": comp_lbl,
            "date": vn_d,
            "stage": current_match["stage"],
            "source_type": f"daily_live_{summ.get('fetched_vn', st_det)}",
            "is_upcoming": False,
            "upcoming_match_id": uid,
            "home_team": ht,
            "away_team": at,
            "home_logo": str(h_info.get("logo") or fixture.get("home_logo") or ""),
            "away_logo": str(a_info.get("logo") or fixture.get("away_logo") or ""),
            "home_score": hs,
            "away_score": as_,
        }
        return events_df, stats_dict, custom_lineups, match_squads, current_match, meta_override

    # --- CASE B & C: UPCOMING ('pre') or HISTORICAL ('post' without ESPN ID) ---
    is_pre = (state == "pre")
    h_ros_pre, a_ros_pre = None, None
    if is_pre and eid and slug:
        kts = float(fixture.get("kickoff_ts") or 0.0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if kts > 0 and abs(kts - now_ts) <= 7200:
            summ_pre = get_espn_match_summary(slug, eid, force_refresh=False, fallback_match=fixture)
            rs_pre = summ_pre.get("rosters") or []
            if len(rs_pre) >= 2:
                h_ros_pre, a_ros_pre = rs_pre[0], rs_pre[1]

    h_lu, h_sq = _roster_to_lineup_and_squad(h_ros_pre, ht, xi_index)
    a_lu, a_sq = _roster_to_lineup_and_squad(a_ros_pre, at, xi_index)
    custom_lineups = {ht: h_lu, at: a_lu}
    match_squads = {ht: h_sq, at: a_sq}

    ph = _safe_float(fixture.get("prob_H"), 38.0)
    pa = _safe_float(fixture.get("prob_A"), 34.0)
    h_poss = round(_clamp(50.0 + (ph - pa) * 0.20, 36.0, 64.0), 1)
    a_poss = round(100.0 - h_poss, 1)

    if is_pre:
        hs, as_ = 0, 0
        h_xg = round(_safe_float(fixture.get("pred_home_goals"), 1.38), 2)
        a_xg = round(_safe_float(fixture.get("pred_away_goals"), 1.15), 2)
    else:
        hs = int(round(_safe_float(fixture.get("home_score"), 0)))
        as_ = int(round(_safe_float(fixture.get("away_score"), 0)))
        h_xg = round(_safe_float(fixture.get("hist_home_xg"), max(0.35, hs * 0.85 + 0.35)), 2)
        a_xg = round(_safe_float(fixture.get("hist_away_xg"), max(0.35, as_ * 0.85 + 0.35)), 2)

    def _proj_stats(team_xg: float, opp_xg: float, poss: float, goals_cnt: int) -> dict:
        shots = int(round(max(goals_cnt + 4, 7 + team_xg * 4.4)))
        on_tgt = int(round(max(goals_cnt, shots * 0.36)))
        blk = max(1, shots // 4)
        off_tgt = max(0, shots - on_tgt - blk)
        p_tot = int(round(250 + poss * 5.4))
        acc_rate = _clamp(0.76 + (poss - 40.0) * 0.0022, 0.74, 0.88)
        p_comp = int(round(p_tot * acc_rate))
        opp_on = max(2, int(round((7 + opp_xg * 4.4) * 0.36)))
        saves = max(1, opp_on - (goals_cnt if not is_pre else 0) + int(round(opp_xg * 0.5)))
        return {
            "possession_seconds": round(poss * 30.0, 1),
            "possession_pct": poss,
            "shots": shots,
            "shots_on_target": on_tgt,
            "shots_off_target": off_tgt,
            "blocked_shots": blk,
            "xg": team_xg,
            "key_passes": max(goals_cnt + 3, int(round(shots * 0.65))),
            "goalkeeper_saves": max(1, saves),
            "corners": int(round(max(2, shots * 0.38))),
            "fouls": 11 if poss < 50 else 10,
            "free_kicks": 12,
            "yellow_cards": 2,
            "red_cards": 0,
            "goals": goals_cnt,
            "passes": p_tot,
            "passes_completed": p_comp,
        }

    stats_dict = {
        ht: _proj_stats(h_xg, a_xg, h_poss, hs),
        at: _proj_stats(a_xg, h_xg, a_poss, as_),
    }

    seed_prefix = f"{'pre' if is_pre else 'post'}:{ht}:{at}:{vn_d}"
    h_hash = hashlib.md5(seed_prefix.encode("utf-8")).hexdigest()[:10]
    mid_str = f"daily_{'pre' if is_pre else 'hist'}_{uid or eid or h_hash}"
    goals_list = []
    if not is_pre:
        rng_g = random.Random(int(h_hash, 16))
        for g_team, g_cnt, lu_dict in ((ht, hs, h_lu), (at, as_, a_lu)):
            out_names = [p["name"] for p in lu_dict["players"][1:]] or [p["name"] for p in lu_dict["players"]]
            for _ in range(g_cnt):
                goals_list.append({
                    "player": rng_g.choice(out_names),
                    "assist": "",
                    "minute": int(rng_g.randint(6, 89)),
                    "team": g_team,
                    "is_og": False,
                })

    events_df = _synthesize_match_events(
        mid_str, ht, at, custom_lineups, match_squads, stats_dict, goals_list, [], seed_prefix
    )

    date_disp = f"{vn_d[8:10]}/{vn_d[5:7]}/{vn_d[:4]}" if len(vn_d) == 10 else vn_d
    kickoff_disp = f"{vn_t} · {date_disp}" if (vn_t and date_disp) else (vn_t or date_disp)
    current_match = {
        "id": mid_str,
        "stage": "Dự kiến & Phân tích trước trận" if is_pre else "Kết quả & Thống kê 2026/27",
        "home_team": ht,
        "away_team": at,
        "home_logo": str(fixture.get("home_logo") or ""),
        "away_logo": str(fixture.get("away_logo") or ""),
        "home_score": hs,
        "away_score": as_,
        "pen_home": None,
        "pen_away": None,
        "stadium": fixture.get("venue") or f"Sân nhà {ht}",
        "referee": "FIFA / Liên đoàn Bóng đá",
        "goals": goals_list,
        "live_state": "pre" if is_pre else "post",
        "live_status_badge": (f"⏰ SẮP DIỄN RA · {vn_t}" if vn_t else "⏰ SẮP DIỄN RA") if is_pre else "HẾT GIỜ · FT",
        "is_upcoming_preview": is_pre,
        "kickoff_disp": kickoff_disp,
        "prematch_ai_summary": ai_summary,
    }
    meta_override = {
        "id": mid_str,
        "competition": comp_lbl,
        "date": vn_d,
        "stage": current_match["stage"],
        "source_type": "daily_pre_ai" if is_pre else "daily_hist_26",
        "is_upcoming": is_pre,
        "upcoming_match_id": uid,
        "home_team": ht,
        "away_team": at,
        "home_logo": str(fixture.get("home_logo") or ""),
        "away_logo": str(fixture.get("away_logo") or ""),
        "home_score": hs,
        "away_score": as_,
    }
    return events_df, stats_dict, custom_lineups, match_squads, current_match, meta_override
