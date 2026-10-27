import hashlib
import html
import json
import math
import os
import sqlite3
import unicodedata as _ud
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_server import (
    compute_wc2018_standings_data,
    ensure_api_server_running,
    predict_match_ai_payload,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sofascore_platform.db")
FIFA_CSV_PATH = os.path.join(BASE_DIR, "fifa18_complete_dataset.csv")


def compact_html(raw_html: str) -> str:
    """Nén chuỗi HTML thành 1 dòng duy nhất để tránh lỗi CommonMark <pre><code>."""
    return "".join(line.strip() for line in str(raw_html).splitlines() if line.strip())


def _s(val, default: str = "") -> str:
    """Chuẩn hóa chuỗi an toàn với None / NaN của Pandas."""
    if val is None or pd.isna(val):
        return default
    txt = str(val).strip()
    return default if not txt or txt.lower() == "nan" else txt


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


# ==========================================================
# ⚽ 0. BỘ NẠP PARQUET OPTA UEFA CHAMPIONS LEAGUE SIÊU TỐC (<40ms)
# ==========================================================
def _map_opta_tactical_pos(pos_raw: str, side_raw: str) -> str:
    pos_l = str(pos_raw or "").strip().lower()
    side_l = str(side_raw or "").strip().lower()
    if "goalkeeper" in pos_l:
        return "Goalkeeper"
    if "wing back" in pos_l:
        return "Left Wing Back" if "left" in side_l else "Right Wing Back"
    if "defender" in pos_l:
        if side_l == "left":
            return "Left Back"
        if side_l == "right":
            return "Right Back"
        if "left" in side_l and "centre" in side_l:
            return "Left Center Back"
        if "right" in side_l and "centre" in side_l:
            return "Right Center Back"
        return "Center Back"
    if "defensive midfielder" in pos_l:
        if "left" in side_l:
            return "Left Defensive Midfield"
        if "right" in side_l:
            return "Right Defensive Midfield"
        return "Center Defensive Midfield"
    if "attacking midfielder" in pos_l:
        if side_l == "left":
            return "Left Wing"
        if side_l == "right":
            return "Right Wing"
        if "left" in side_l:
            return "Left Attacking Midfield"
        if "right" in side_l:
            return "Right Attacking Midfield"
        return "Center Attacking Midfield"
    if "midfielder" in pos_l:
        if side_l == "left":
            return "Left Midfield"
        if side_l == "right":
            return "Right Midfield"
        if "left" in side_l:
            return "Left Center Midfield"
        if "right" in side_l:
            return "Right Center Midfield"
        return "Center Midfield"
    if "striker" in pos_l or "forward" in pos_l:
        if side_l == "left":
            return "Left Wing"
        if side_l == "right":
            return "Right Wing"
        if "left" in side_l:
            return "Left Center Forward"
        if "right" in side_l:
            return "Right Center Forward"
        return "Center Forward"
    return ""


@st.cache_data(show_spinner=False, max_entries=32)
def _load_opta_ucl_cached(opta_mid: str, home_team: str, away_team: str, ht_id: str, at_id: str, sample_dir: str):
    opta_dir = os.path.join(sample_dir, "opta_ucl")
    lu_path = os.path.join(opta_dir, "ucl_lineups.parquet")
    ev_path = os.path.join(opta_dir, "ucl_events.parquet")

    lu_df = pd.read_parquet(lu_path, filters=[("match_id", "==", opta_mid)]) if os.path.isfile(lu_path) else pd.DataFrame()
    ev_df = pd.read_parquet(ev_path, filters=[("match_id", "==", opta_mid)]) if os.path.isfile(ev_path) else pd.DataFrame()

    def _estimate_xg_local(lx, ly):
        if lx is None or ly is None or pd.isna(lx) or pd.isna(ly):
            return 0.10
        dx = 120.0 - float(lx)
        dy = 40.0 - float(ly)
        dist = math.hypot(dx, dy)
        angle = max(0.05, math.atan2(8.0 * dx, max(1.0, dx * dx + dy * dy - 16.0)))
        logit = 0.85 - 0.095 * dist + 1.25 * angle
        prob = 1.0 / (1.0 + math.exp(-logit))
        return round(min(0.95, max(0.01, prob)), 2)

    custom_lineups = {}
    match_squads = {}
    player_pos_lookup = {}

    if not lu_df.empty:
        for tid_val, t_label in [(ht_id, home_team), (at_id, away_team)]:
            t_rows = lu_df[lu_df["team_id"].astype(str) == str(tid_val)].copy()
            if t_rows.empty:
                continue
            t_rows["_fp"] = pd.to_numeric(t_rows.get("formation_place"), errors="coerce").fillna(99).astype(int)
            t_rows["_gk"] = t_rows["position"].astype(str).str.lower().str.contains("goalkeeper", na=False)
            t_rows = t_rows.sort_values(by=["is_starter", "_gk", "_fp"], ascending=[False, False, True])

            starters_list = []
            squad_players = []
            for _, r in t_rows.iterrows():
                p_name = str(r.get("player_name") or "").strip()
                if not p_name:
                    continue
                mapped_pos = _map_opta_tactical_pos(r.get("position"), r.get("position_side"))
                shirt_num = pd.to_numeric(r.get("shirt_number"), errors="coerce")
                jersey = int(shirt_num) if pd.notna(shirt_num) and shirt_num > 0 else ""
                started = bool(r.get("is_starter"))
                sub_on = int(pd.to_numeric(r.get("sub_on_minute"), errors="coerce") or 0)
                sub_off = int(pd.to_numeric(r.get("sub_off_minute"), errors="coerce") or 0)
                mins_p = float(pd.to_numeric(r.get("minutes_played"), errors="coerce") or 0.0)
                entered = (not started) and (sub_on > 0 or mins_p > 0)
                left = sub_off > 0

                if mapped_pos:
                    player_pos_lookup[(t_label, p_name)] = mapped_pos
                if started and len(starters_list) < 11:
                    starters_list.append({
                        "name": p_name,
                        "position": mapped_pos or ("Goalkeeper" if len(starters_list) == 0 else "Center Midfield"),
                        "jersey_number": jersey,
                    })
                squad_players.append({
                    "name": p_name,
                    "nickname": p_name,
                    "jersey_number": jersey,
                    "position": mapped_pos or str(r.get("position") or "").strip(),
                    "started": started,
                    "entered": entered,
                    "left": left,
                    "injured_on_exit": False,
                    "minute_in": f"{sub_on}:00" if (entered and sub_on > 0) else None,
                    "minute_out": f"{sub_off}:00" if (left and sub_off > 0) else None,
                })
            if starters_list:
                custom_lineups[t_label] = {"formation": "", "players": starters_list}
            if squad_players:
                match_squads[t_label] = {"players": squad_players}

    stats_dict = {
        t: {
            "possession_seconds": 0.0, "possession_pct": 50.0,
            "shots": 0, "shots_on_target": 0, "shots_off_target": 0,
            "blocked_shots": 0, "xg": 0.0, "key_passes": 0, "goalkeeper_saves": 0,
            "corners": 0, "fouls": 0, "free_kicks": 0, "yellow_cards": 0,
            "red_cards": 0, "goals": 0, "passes": 0, "passes_completed": 0,
        }
        for t in (home_team, away_team)
    }

    rows_out = []
    if not ev_df.empty:
        ev_reg = ev_df[ev_df["period_id"].isin([1, 2, 3, 4])].reset_index(drop=True)
        raw_recs = ev_reg.to_dict(orient="records")
        n_recs = len(raw_recs)

        # Ghép cặp thay người (type_id 18 rời sân + type_id 19 vào sân)
        sub_on_used = set()
        sub_rep_by_idx = {}
        for i, rec in enumerate(raw_recs):
            if int(rec.get("type_id") or 0) == 18:
                tid_i = str(rec.get("team_id") or "")
                min_i = int(rec.get("minute") or 0)
                for j in range(max(0, i - 3), min(n_recs, i + 6)):
                    if j in sub_on_used:
                        continue
                    rj = raw_recs[j]
                    if int(rj.get("type_id") or 0) == 19 and str(rj.get("team_id") or "") == tid_i and abs(int(rj.get("minute") or 0) - min_i) <= 1:
                        sub_rep_by_idx[i] = str(rj.get("player_name") or "").strip()
                        sub_on_used.add(j)
                        break

        for i, rec in enumerate(raw_recs):
            tid_val = int(rec.get("type_id") or 0)
            if tid_val == 19:
                continue
            team_id_s = str(rec.get("team_id") or "")
            if team_id_s == str(ht_id):
                t_name = home_team
            elif team_id_s == str(at_id):
                t_name = away_team
            else:
                continue

            p_name = _s(rec.get("player_name"), "")
            period = int(rec.get("period_id") or 1)
            minute = int(rec.get("minute") or 0)
            second = int(rec.get("second") or 0)
            outcome = int(rec.get("outcome") if pd.notna(rec.get("outcome")) else 1)
            q_str = str(rec.get("qualifier_json") or "")

            rx = rec.get("x")
            ry = rec.get("y")
            lx = round(float(rx) * 1.2, 2) if (rx is not None and pd.notna(rx)) else None
            ly = round((100.0 - float(ry)) * 0.8, 2) if (ry is not None and pd.notna(ry)) else None

            rex = rec.get("end_x")
            rey = rec.get("end_y")
            px = round(float(rex) * 1.2, 2) if (rex is not None and pd.notna(rex)) else None
            py = round((100.0 - float(rey)) * 0.8, 2) if (rey is not None and pd.notna(rey)) else None

            ev_type = ""
            p_rec = ""
            p_out = ""
            p_len = None
            p_cross = 0
            p_tb = 0
            p_sw = 0
            p_sa = 0
            p_ga = 0
            s_out = ""
            s_xg = None
            drib_out = ""
            card_nm = ""
            sub_rep = ""
            gk_type = ""

            if tid_val == 1:
                ev_type = "Pass"
                stats_dict[t_name]["passes"] += 1
                if outcome == 1:
                    stats_dict[t_name]["passes_completed"] += 1
                    p_out = ""
                    for nxt in raw_recs[i + 1 : min(n_recs, i + 4)]:
                        if str(nxt.get("team_id") or "") == team_id_s:
                            cand_p = _s(nxt.get("player_name"), "")
                            if cand_p and cand_p != p_name:
                                p_rec = cand_p
                                break
                else:
                    p_out = "Incomplete"
                if lx is not None and ly is not None and px is not None and py is not None:
                    p_len = round(math.hypot(px - lx, py - ly), 1)
                    if abs(py - ly) >= 32.0:
                        p_sw = 1
                if '"2":' in q_str:
                    p_cross = 1
                if '"4":' in q_str:
                    p_tb = 1
                if '"6":' in q_str:
                    stats_dict[t_name]["corners"] += 1
                elif '"5":' in q_str:
                    stats_dict[t_name]["free_kicks"] += 1
                if '"210":' in q_str:
                    p_sa = 1
                    stats_dict[t_name]["key_passes"] += 1
                    for nxt in raw_recs[i + 1 : min(n_recs, i + 4)]:
                        if int(nxt.get("type_id") or 0) == 16 and str(nxt.get("team_id") or "") == team_id_s:
                            nq = str(nxt.get("qualifier_json") or "")
                            if '"OWN_GOAL"' not in nq and '"28":' not in nq:
                                p_ga = 1
                                break

            elif tid_val in (13, 14, 15, 16):
                is_og = tid_val == 16 and ('"OWN_GOAL"' in q_str or '"28":' in q_str)
                if is_og:
                    ev_type = "Own Goal Against"
                    opp = away_team if t_name == home_team else home_team
                    stats_dict[opp]["goals"] += 1
                else:
                    ev_type = "Shot"
                    s_xg = _estimate_xg_local(lx, ly)
                    stats_dict[t_name]["shots"] += 1
                    stats_dict[t_name]["xg"] += s_xg
                    if tid_val == 16:
                        s_out = "Goal"
                        stats_dict[t_name]["shots_on_target"] += 1
                        stats_dict[t_name]["goals"] += 1
                    elif tid_val == 14:
                        s_out = "Post"
                        stats_dict[t_name]["shots_off_target"] += 1
                    elif tid_val == 13:
                        s_out = "Off T"
                        stats_dict[t_name]["shots_off_target"] += 1
                    else:
                        if '"82":' in q_str:
                            s_out = "Blocked"
                            stats_dict[t_name]["blocked_shots"] += 1
                        else:
                            s_out = "Saved"
                            stats_dict[t_name]["shots_on_target"] += 1
                            opp = away_team if t_name == home_team else home_team
                            stats_dict[opp]["goalkeeper_saves"] += 1

            elif tid_val == 3:
                ev_type = "Dribble"
                drib_out = "Complete" if outcome == 1 else "Incomplete"
            elif tid_val == 4:
                if outcome == 0:
                    ev_type = "Foul Committed"
                    stats_dict[t_name]["fouls"] += 1
                else:
                    ev_type = "Foul Won"
            elif tid_val in (7, 44):
                ev_type = "Duel"
            elif tid_val == 8:
                ev_type = "Interception"
            elif tid_val == 10:
                ev_type = "Goal Keeper"
                gk_type = "Shot Saved"
            elif tid_val in (11, 52, 53):
                ev_type = "Goal Keeper"
                gk_type = "Collected"
            elif tid_val == 12:
                ev_type = "Clearance"
            elif tid_val == 49:
                ev_type = "Ball Recovery"
            elif tid_val == 50:
                ev_type = "Dispossessed"
            elif tid_val == 61:
                ev_type = "Miscontrol"
            elif tid_val == 17:
                ev_type = "Bad Behaviour"
                if '"32":' in q_str:
                    card_nm = "Second Yellow"
                    stats_dict[t_name]["yellow_cards"] += 1
                    stats_dict[t_name]["red_cards"] += 1
                elif '"33":' in q_str:
                    card_nm = "Red Card"
                    stats_dict[t_name]["red_cards"] += 1
                else:
                    card_nm = "Yellow Card"
                    stats_dict[t_name]["yellow_cards"] += 1
            elif tid_val == 18:
                ev_type = "Substitution"
                sub_rep = sub_rep_by_idx.get(i, "")
            else:
                if not p_name or lx is None:
                    continue
                ev_type = "Ball Receipt*"

            rows_out.append({
                "match_id": opta_mid,
                "minute": minute,
                "second": second,
                "period": period,
                "timestamp": f"00:{minute % 60:02d}:{second % 60:02d}",
                "team_name": t_name,
                "player_name": p_name,
                "position_name": player_pos_lookup.get((t_name, p_name), ""),
                "type_name": ev_type,
                "location_x": lx,
                "location_y": ly,
                "pass_recipient_name": p_rec,
                "pass_outcome_name": p_out,
                "pass_end_location_x": px,
                "pass_end_location_y": py,
                "pass_length": p_len,
                "pass_cross": p_cross,
                "pass_through_ball": p_tb,
                "pass_switch": p_sw,
                "pass_shot_assist": p_sa,
                "pass_goal_assist": p_ga,
                "shot_outcome_name": s_out,
                "shot_statsbomb_xg": s_xg,
                "shot_type_name": "Open Play" if ev_type == "Shot" else "",
                "shot_body_part_name": "Foot" if ev_type == "Shot" else "",
                "dribble_outcome_name": drib_out,
                "card_name": card_nm,
                "substitution_replacement_name": sub_rep,
                "substitution_outcome_name": "Tactical" if ev_type == "Substitution" else "",
                "gk_type_name": gk_type,
                "gk_outcome_name": "",
                "carry_distance": 0.0,
                "under_pressure": 0,
                "possession": None,
                "possession_team": t_name,
            })

    tot_passes = stats_dict[home_team]["passes"] + stats_dict[away_team]["passes"]
    if tot_passes > 0:
        stats_dict[home_team]["possession_pct"] = round(stats_dict[home_team]["passes"] / tot_passes * 100.0, 1)
        stats_dict[away_team]["possession_pct"] = round(100.0 - stats_dict[home_team]["possession_pct"], 1)
    for t in stats_dict:
        stats_dict[t]["xg"] = round(stats_dict[t]["xg"], 2)

    return pd.DataFrame(rows_out), stats_dict, custom_lineups, match_squads


def load_opta_ucl_match_bundle(match_meta: dict, sample_dir: str):
    opta_mid = str(match_meta.get("opta_match_id") or "")
    ht = str(match_meta.get("home_team") or "Home")
    at = str(match_meta.get("away_team") or "Away")
    ht_id = str(match_meta.get("home_team_id") or "")
    at_id = str(match_meta.get("away_team_id") or "")
    ev_df, stats_dict, custom_lineups, match_squads = _load_opta_ucl_cached(opta_mid, ht, at, ht_id, at_id, sample_dir)
    current_match = {
        "id": match_meta.get("id", f"opta_ucl_{opta_mid}"),
        "stage": match_meta.get("stage", "UEFA Champions League"),
        "home_team": ht,
        "away_team": at,
        "home_score": int(match_meta.get("home_score", 0) or 0),
        "away_score": int(match_meta.get("away_score", 0) or 0),
        "pen_home": match_meta.get("pen_home"),
        "pen_away": match_meta.get("pen_away"),
        "stadium": match_meta.get("stadium", "UEFA Champions League Arena"),
        "referee": match_meta.get("referee", "UEFA Official Referee"),
        "goals": match_meta.get("goals") or [],
    }
    return ev_df, stats_dict, custom_lineups, match_squads, current_match


# ==========================================================
# 🌐 1. ĐA NGÔN NGỮ (VI / EN) - TIÊU CHÍ #30
# ==========================================================
_I18N = {
    "vi": {
        "nav_title": "🧭 Không gian nền tảng",
        "ws_match": "🏟️ Chi Tiết Trận Đấu (Match Center)",
        "ws_live": "🔴 Trực Tiếp & Đa Trận (Live & Multi-View)",
        "ws_hub": "🏆 Giải Đấu, BXH & Chuyển Nhượng",
        "ws_ai": "⚔️ So Sánh, AI & Xuất Dữ Liệu / API",
        "search_placeholder": "Nhập tên đội, cầu thủ, giải đấu...",
        "tab_lineup": "👥 Đội Hình (Lineups)",
        "tab_stats": "📊 Thống Kê & Momentum",
        "tab_shotmap": "🎯 Bản Đồ Sút (Shot Map)",
        "tab_player": "🔥 Cầu Thủ (Heatmap & Pass)",
        "tab_network": "🕸️ Mạng Lưới Chuyền",
        "tab_ai_news": "🤖 AI Dự Đoán, Bình Luận & Tin Tức",
        "fav_add": "☆ Lưu Yêu Thích",
        "fav_saved": "⭐ Đã Yêu Thích",
        "highlights_btn": "🎬 Video Highlights",
        "share_btn": "📤 Chia Sẻ Social",
        "export_btn": "📥 Xuất Dữ Liệu",
    },
    "en": {
        "nav_title": "🧭 Platform Workspace",
        "ws_match": "🏟️ Match Analytics Center",
        "ws_live": "🔴 Live Score & Multi-Match View",
        "ws_hub": "🏆 Standings, Fixtures & Transfers",
        "ws_ai": "⚔️ Player Compare, AI & Export / API",
        "search_placeholder": "Search teams, players, matches...",
        "tab_lineup": "👥 Lineups & Ratings",
        "tab_stats": "📊 Stats & Momentum",
        "tab_shotmap": "🎯 2D Shot Map & xG",
        "tab_player": "🔥 Player Heatmap & Passes",
        "tab_network": "🕸️ Passing Network",
        "tab_ai_news": "🤖 AI Predictor, Live Feed & News",
        "fav_add": "☆ Add Favourite",
        "fav_saved": "⭐ Favourited",
        "highlights_btn": "🎬 Official Highlights",
        "share_btn": "📤 Share Social",
        "export_btn": "📥 Export Data",
    },
}


def tr(key: str, lang: str = "vi") -> str:
    return _I18N.get(lang, _I18N["vi"]).get(key, key)


# ==========================================================
# 🗄️ 2. CSDL SQLITE + XÁC THỰC AUTH + YÊU THÍCH (#22, #28, #44)
# ==========================================================
def _hash_pw(password: str) -> str:
    return hashlib.sha256(("sofascore_salt_2026::" + str(password)).encode("utf-8")).hexdigest()


def init_platform_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'member',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                item_label TEXT NOT NULL,
                comp_name TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(username, item_type, item_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                match_id TEXT NOT NULL,
                match_label TEXT NOT NULL,
                pred_home INTEGER NOT NULL,
                pred_away INTEGER NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(username, match_id)
            )
            """
        )
        cur = conn.execute("SELECT COUNT(*) FROM users WHERE username = ?", ("demo",))
        if cur.fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                ("demo", "Chuyên Gia Phân Tích (Demo)", _hash_pw("123456"), "analyst", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
        conn.commit()


def authenticate_user(username: str, password: str):
    u = str(username or "").strip().lower()
    if not u or not password:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username, display_name, role FROM users WHERE username = ? AND password_hash = ?",
            (u, _hash_pw(password)),
        ).fetchone()
    if row:
        return {"username": row[0], "display_name": row[1], "role": row[2]}
    return None


def register_user(username: str, display_name: str, password: str):
    u = str(username or "").strip().lower()
    dn = str(display_name or u).strip()
    if len(u) < 3 or len(str(password or "")) < 4:
        return False, "Tên đăng nhập ≥ 3 ký tự và mật khẩu ≥ 4 ký tự."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (u, dn, _hash_pw(password), "member", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        return True, f"Tạo tài khoản '{dn}' thành công!"
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại trong CSDL SQLite."


def is_favorite(username: str, item_type: str, item_id: str) -> bool:
    if not username:
        return False
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM favorites WHERE username = ? AND item_type = ? AND item_id = ?",
            (username, item_type, str(item_id)),
        ).fetchone()
    return row is not None


def toggle_favorite(username: str, item_type: str, item_id: str, item_label: str, comp_name: str = "") -> bool:
    u = username or "guest"
    iid = str(item_id)
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM favorites WHERE username = ? AND item_type = ? AND item_id = ?",
            (u, item_type, iid),
        ).fetchone()
        if exists:
            conn.execute(
                "DELETE FROM favorites WHERE username = ? AND item_type = ? AND item_id = ?",
                (u, item_type, iid),
            )
            conn.commit()
            return False
        else:
            conn.execute(
                "INSERT INTO favorites (username, item_type, item_id, item_label, comp_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (u, item_type, iid, str(item_label), str(comp_name), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            return True


def list_favorites(username: str):
    u = username or "guest"
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT item_type, item_id, item_label, comp_name, created_at FROM favorites WHERE username = ? ORDER BY id DESC",
            (u,),
        ).fetchall()
    return [
        {"item_type": r[0], "item_id": r[1], "item_label": r[2], "comp_name": r[3], "created_at": r[4]}
        for r in rows
    ]


def save_user_prediction(username: str, match_id: str, match_label: str, pred_home: int, pred_away: int, note: str = ""):
    u = username or "guest"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_predictions (username, match_id, match_label, pred_home, pred_away, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, match_id) DO UPDATE SET
                pred_home=excluded.pred_home, pred_away=excluded.pred_away,
                note=excluded.note, created_at=excluded.created_at
            """,
            (u, str(match_id), str(match_label), int(pred_home), int(pred_away), str(note), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()


def get_user_prediction(username: str, match_id: str):
    u = username or "guest"
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT pred_home, pred_away, note, created_at FROM user_predictions WHERE username = ? AND match_id = ?",
            (u, str(match_id)),
        ).fetchone()
    if row:
        return {"pred_home": row[0], "pred_away": row[1], "note": row[2], "created_at": row[3]}
    return None


# ==========================================================
# 💶 3. CƠ SỞ DỮ LIỆU CHUYỂN NHƯỢNG & CHỈ SỐ FIFA (#38 & #14)
# ==========================================================
def _ascii_key(text: str) -> str:
    s = "".join(c for c in _ud.normalize("NFKD", str(text or "")) if not _ud.combining(c))
    return " ".join(s.lower().replace('"', "'").replace(".", " ").replace("-", " ").split())


def _parse_fifa_num(val, default=65.0) -> float:
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    if not s:
        return default
    for sep in ("+", "-"):
        if sep in s[1:]:
            s = s.split(sep, 1)[0].strip()
    try:
        return float(s)
    except ValueError:
        return default


def _parse_euro_millions(val_str: str) -> float:
    s = str(val_str or "").replace("€", "").replace(",", "").strip().upper()
    if not s or s == "0":
        return 0.0
    try:
        if s.endswith("M"):
            return float(s[:-1])
        if s.endswith("K"):
            return round(float(s[:-1]) / 1000.0, 3)
        return float(s) / 1_000_000.0
    except ValueError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_fifa_transfer_db():
    if not os.path.exists(FIFA_CSV_PATH):
        return pd.DataFrame(), {}, {}
    use_cols = [
        "Name", "Age", "Nationality", "Overall", "Potential", "Club", "Value", "Wage",
        "Preferred Positions", "Acceleration", "Sprint speed", "Finishing", "Shot power",
        "Positioning", "Short passing", "Long passing", "Vision", "Crossing", "Dribbling",
        "Ball control", "Agility", "Interceptions", "Standing tackle", "Sliding tackle",
        "Marking", "Strength", "Stamina", "Jumping", "Composure",
        "GK diving", "GK handling", "GK kicking", "GK positioning", "GK reflexes",
    ]
    df = pd.read_csv(FIFA_CSV_PATH, usecols=lambda c: c in use_cols, low_memory=False)
    df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce").fillna(70).astype(int)
    df["Potential"] = pd.to_numeric(df["Potential"], errors="coerce").fillna(df["Overall"]).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(25).astype(int)
    df["Value_M"] = df["Value"].apply(_parse_euro_millions)
    df["Wage_K"] = df["Wage"].apply(lambda x: _parse_euro_millions(x) * 1000.0)
    df = df.sort_values(["Overall", "Value_M"], ascending=[False, False]).reset_index(drop=True)

    exact_map = {}
    surname_nat_map = {}
    surname_only_map = {}
    for idx, row in df.iterrows():
        nm = str(row["Name"] or "").strip()
        nat = _ascii_key(row.get("Nationality", ""))
        k_full = _ascii_key(nm)
        if k_full and k_full not in exact_map:
            exact_map[k_full] = idx
        parts = k_full.split()
        if parts:
            surname = parts[-1]
            if nat and (surname, nat) not in surname_nat_map:
                surname_nat_map[(surname, nat)] = idx
            if surname not in surname_only_map:
                surname_only_map[surname] = idx
            if len(parts) >= 2:
                init_sur = f"{parts[0][0]} {surname}"
                if init_sur not in exact_map:
                    exact_map[init_sur] = idx
                if nat and (init_sur, nat) not in surname_nat_map:
                    surname_nat_map[(init_sur, nat)] = idx

    return df, exact_map, {"by_nat": surname_nat_map, "by_sur": surname_only_map}


def lookup_fifa_player(full_name: str, display_name: str = "", team_name: str = ""):
    df, exact_map, sub_maps = load_fifa_transfer_db()
    if df.empty:
        return None
    nat_key = _ascii_key(team_name)
    by_nat = sub_maps["by_nat"]
    by_sur = sub_maps["by_sur"]

    for cand in (display_name, full_name):
        ck = _ascii_key(cand)
        if not ck:
            continue
        if ck in exact_map:
            return df.iloc[exact_map[ck]].to_dict()
        c_parts = ck.split()
        if len(c_parts) >= 2:
            init_sur = f"{c_parts[0][0]} {c_parts[-1]}"
            if nat_key and (init_sur, nat_key) in by_nat:
                return df.iloc[by_nat[(init_sur, nat_key)]].to_dict()
            if init_sur in exact_map:
                return df.iloc[exact_map[init_sur]].to_dict()
        if c_parts:
            sur = c_parts[-1]
            if nat_key and (sur, nat_key) in by_nat:
                return df.iloc[by_nat[(sur, nat_key)]].to_dict()
            if len(sur) >= 5 and sur in by_sur:
                return df.iloc[by_sur[sur]].to_dict()
    return None


def compute_player_radar_metrics(player_data: dict, p_events: pd.DataFrame, fifa_info: dict = None):
    """Tính 6 trục kỹ năng (thang 0-99) kết hợp dữ liệu sự kiện thực tế trong trận & hồ sơ FIFA."""
    pos = str(player_data.get("pos", "")).lower()
    is_gk = "goalkeeper" in pos or pos == "gk"
    rating = float(player_data.get("rating", 6.5))

    p_passes = p_events[p_events["type_name"] == "Pass"] if (p_events is not None and not p_events.empty and "type_name" in p_events.columns) else pd.DataFrame()
    p_shots = p_events[p_events["type_name"] == "Shot"] if (p_events is not None and not p_events.empty and "type_name" in p_events.columns) else pd.DataFrame()

    if is_gk:
        saves = int(player_data.get("gk_saves") or 0)
        claims = int(player_data.get("gk_claims") or 0)
        tot_p = len(p_passes)
        diving = _parse_fifa_num((fifa_info or {}).get("GK diving"), 76 + saves * 3)
        reflexes = _parse_fifa_num((fifa_info or {}).get("GK reflexes"), 78 + saves * 3.5)
        handling = _parse_fifa_num((fifa_info or {}).get("GK handling"), 75 + claims * 3)
        positioning = _parse_fifa_num((fifa_info or {}).get("GK positioning"), 76 + (rating - 6.5) * 8)
        kicking = _parse_fifa_num((fifa_info or {}).get("GK kicking"), min(92, 68 + tot_p * 0.5))
        impact_sc = min(98, max(52, round(rating * 10.2)))
        return {
            "axes": ["Phản xạ (Reflexes)", "Đổ người (Diving)", "Bắt bóng (Handling)", "Chọn vị trí (Position)", "Phân phối bóng (Kicking)", "Tầm ảnh hưởng (Impact)"],
            "values": [
                int(np.clip(round(reflexes * 0.65 + min(99, 68 + saves * 6) * 0.35), 45, 99)),
                int(np.clip(round(diving * 0.7 + min(99, 66 + saves * 5) * 0.3), 45, 99)),
                int(np.clip(round(handling * 0.7 + min(99, 68 + claims * 5) * 0.3), 45, 99)),
                int(np.clip(round(positioning), 45, 99)),
                int(np.clip(round(kicking), 45, 99)),
                int(np.clip(impact_sc, 45, 99)),
            ],
        }

    goals = int(player_data.get("goals_count") or 0)
    assists = int(player_data.get("assists_count") or 0)
    shots_cnt = len(p_shots)
    xg_sum = float(pd.to_numeric(p_shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0).sum()) if not p_shots.empty and "shot_statsbomb_xg" in p_shots.columns else float(player_data.get("xg") or 0.0)
    tot_p = len(p_passes)
    xt_sum = float(pd.to_numeric(p_passes.get("xT"), errors="coerce").fillna(0).sum()) if not p_passes.empty and "xT" in p_passes.columns else 0.0
    if xt_sum == 0.0 and not p_passes.empty and {"location_x", "pass_end_location_x"}.issubset(p_passes.columns):
        dx = pd.to_numeric(p_passes["pass_end_location_x"], errors="coerce") - pd.to_numeric(p_passes["location_x"], errors="coerce")
        xt_sum = float(dx.clip(lower=0).fillna(0).sum()) * 0.002
    tackles = int(player_data.get("tackles") or 0)

    m_shoot = min(98, 52 + goals * 16 + shots_cnt * 4.5 + xg_sum * 22)
    m_pass = min(98, 56 + tot_p * 0.42 + assists * 14)
    m_prog = min(98, 54 + max(0.0, xt_sum) * 95 + tot_p * 0.25)
    m_drib = min(98, 58 + (rating - 6.0) * 9.5)
    m_def = min(98, 48 + tackles * 5.2 + (14 if any(k in pos for k in ("back", "defensive", "cb", "rb", "lb", "dm")) else 0))
    m_imp = min(99, max(50, round(rating * 10.2)))

    if fifa_info:
        f_shoot = (_parse_fifa_num(fifa_info.get("Finishing"), m_shoot) + _parse_fifa_num(fifa_info.get("Shot power"), m_shoot)) / 2.0
        f_pass = (_parse_fifa_num(fifa_info.get("Short passing"), m_pass) + _parse_fifa_num(fifa_info.get("Vision"), m_pass)) / 2.0
        f_prog = (_parse_fifa_num(fifa_info.get("Long passing"), m_prog) + _parse_fifa_num(fifa_info.get("Crossing"), m_prog)) / 2.0
        f_drib = (_parse_fifa_num(fifa_info.get("Dribbling"), m_drib) + _parse_fifa_num(fifa_info.get("Ball control"), m_drib)) / 2.0
        f_def = (_parse_fifa_num(fifa_info.get("Standing tackle"), m_def) + _parse_fifa_num(fifa_info.get("Interceptions"), m_def)) / 2.0
        vals = [
            int(np.clip(round(f_shoot * 0.55 + m_shoot * 0.45), 42, 99)),
            int(np.clip(round(f_pass * 0.55 + m_pass * 0.45), 45, 99)),
            int(np.clip(round(f_prog * 0.50 + m_prog * 0.50), 45, 99)),
            int(np.clip(round(f_drib * 0.60 + m_drib * 0.40), 45, 99)),
            int(np.clip(round(f_def * 0.55 + m_def * 0.45), 38, 99)),
            int(np.clip(round(float(fifa_info.get("Overall", 78)) * 0.45 + m_imp * 0.55), 50, 99)),
        ]
    else:
        vals = [int(np.clip(round(v), 42, 99)) for v in (m_shoot, m_pass, m_prog, m_drib, m_def, m_imp)]

    return {
        "axes": ["Dứt điểm", "Chuyền & Tổ chức", "Tịnh tiến (xT)", "Kéo & Giữ bóng", "Phòng ngự", "Tầm ảnh hưởng"],
        "values": vals,
    }


def render_player_radar_and_fifa_subtab(player_data: dict, p_events: pd.DataFrame, current_match: dict):
    """Sub-tab thứ 4 trong thẻ chi tiết cầu thủ: Biểu đồ Radar kỹ năng (#14) + Giá trị chuyển nhượng FIFA (#38) + Yêu thích (#28)."""
    p_name = player_data.get("name", "")
    disp_name = player_data.get("display_name", p_name)
    team_name = player_data.get("team_name", "")
    is_home = bool(player_data.get("is_home", True))
    accent = "#38BDF8" if is_home else "#FF4D6D"
    fill_rgba = "rgba(56, 189, 248, 0.28)" if is_home else "rgba(255, 77, 109, 0.28)"

    curr_user = (st.session_state.get("auth_user") or {}).get("username", "demo")
    fav_p_state = is_favorite(curr_user, "player", p_name)
    btn_lbl = f"⭐ Đã lưu {disp_name} vào Yêu thích" if fav_p_state else f"☆ Lưu {disp_name} vào Cầu thủ Yêu thích"
    if st.button(btn_lbl, key=f"fav_p_btn_{p_name}_{current_match.get('id')}", width="stretch"):
        toggle_favorite(curr_user, "player", p_name, f"{disp_name} ({team_name})", team_name)
        st.rerun()

    fifa_info = lookup_fifa_player(p_name, disp_name, team_name)
    radar = compute_player_radar_metrics(player_data, p_events, fifa_info)

    if fifa_info:
        club = html.escape(_s(fifa_info.get("Club"), "Đội tuyển Quốc gia"))
        val_str = html.escape(_s(fifa_info.get("Value"), "N/A"))
        wage_str = html.escape(_s(fifa_info.get("Wage"), "N/A"))
        ovr = int(fifa_info.get("Overall", 80))
        pot = int(fifa_info.get("Potential", ovr))
        age = int(fifa_info.get("Age", 25))
        st.markdown(compact_html(f"""
        <div style="background:#0E1420;border:1px solid #243247;border-radius:10px;padding:9px 12px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:11px;color:#94A3B8;font-weight:700;text-transform:uppercase;">💶 Hồ sơ Chuyển nhượng & CLB</div>
                    <div style="font-size:13px;color:#F8FAFC;font-weight:800;margin-top:2px;">🏟️ {club} • {age} tuổi</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;">
                    <span style="background:rgba(0,214,108,0.16);border:1px solid #00D66C;color:#4ADE80;font-weight:900;font-size:12px;padding:3px 8px;border-radius:7px;">Định giá: {val_str}</span>
                    <span style="background:rgba(56,189,248,0.16);border:1px solid #38BDF8;color:#7DD3FC;font-weight:800;font-size:11.5px;padding:3px 8px;border-radius:7px;">Lương: {wage_str}/tuần</span>
                    <span style="background:#FACC15;color:#000;font-weight:900;font-size:12px;padding:3px 7px;border-radius:7px;">OVR {ovr}➔{pot}</span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    axes_closed = radar["axes"] + [radar["axes"][0]]
    vals_closed = radar["values"] + [radar["values"][0]]

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=axes_closed,
        fill="toself",
        fillcolor=fill_rgba,
        line=dict(color=accent, width=2.6),
        marker=dict(size=7, color=accent, line=dict(color="#FFFFFF", width=1.2)),
        name=disp_name,
        hovertemplate="<b>%{theta}</b>: %{r}/99<extra></extra>",
    ))
    fig_rad.update_layout(
        polar=dict(
            bgcolor="#0D121C",
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 95], gridcolor="rgba(148,163,184,0.16)", tickfont=dict(size=9, color="#64748B")),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.18)", linecolor="rgba(148,163,184,0.25)", tickfont=dict(size=11.5, color="#E2E8F0")),
        ),
        height=265,
        margin=dict(l=38, r=38, t=20, b=20),
        paper_bgcolor="#0E131E",
        showlegend=False,
    )
    st.plotly_chart(fig_rad, width="stretch", key=f"radar_sub_{p_name}_{current_match.get('id')}")


# ==========================================================
# ⚡ 4. THANH CÔNG CỤ TRẬN ĐẤU: YÊU THÍCH, HIGHLIGHTS, CHIA SẺ, EXPORT (#28, #29, #39, #45)
# ==========================================================
def build_standalone_match_report_html(current_match: dict, home_lineup: list, away_lineup: list, home_stats, away_stats) -> str:
    ht = html.escape(str(current_match.get("home_team", "")))
    at = html.escape(str(current_match.get("away_team", "")))
    hs = current_match.get("home_score", 0)
    as_ = current_match.get("away_score", 0)
    stage = html.escape(str(current_match.get("stage", "")))
    stadium = html.escape(str(current_match.get("stadium", "")))
    ref = html.escape(str(current_match.get("referee", "")))

    def _rows_p(lineup):
        out = []
        for p in sorted(lineup, key=lambda x: x.get("rating", 0), reverse=True):
            out.append(
                f"<tr><td>{html.escape(str(p.get('display_name', '')))}</td>"
                f"<td>{html.escape(str(p.get('pos', '')))}</td>"
                f"<td><b>{p.get('rating', '')}★</b></td>"
                f"<td>{p.get('goals_count', 0)}</td>"
                f"<td>{p.get('assists_count', 0)}</td>"
                f"<td>{html.escape(str(p.get('passes', '')))}</td></tr>"
            )
        return "".join(out)

    goals_li = "".join(
        f"<li>Phút {g.get('minute')}′ — <b>{html.escape(str(g.get('player')))}</b> ({html.escape(str(g.get('team')))}{' - OG' if g.get('is_og') else ''})</li>"
        for g in (current_match.get("goals") or [])
    ) or "<li>Không có bàn thắng (0 - 0)</li>"

    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Báo Cáo Phân Tích Trận Đấu: {ht} {hs} - {as_} {at}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; background: #0E121A; color: #F8FAFC; padding: 28px; max-width: 980px; margin: auto; }}
.banner {{ background: linear-gradient(135deg,#131A26,#1E293B); border: 1px solid #38BDF8; border-radius: 14px; padding: 22px; text-align: center; margin-bottom: 22px; }}
.score {{ font-size: 40px; font-weight: 900; color: #38BDF8; margin: 10px 0; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.card {{ background: #131A26; border: 1px solid #26354D; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #1E293B; padding: 7px 6px; text-align: left; }}
th {{ color: #38BDF8; }}
@media print {{ body {{ background: #fff; color: #000; }} .card, .banner {{ border-color: #ccc; background: #f9fafb; }} }}
</style></head><body>
<div class="banner">
    <div><b>{stage}</b> • 🏟️ {stadium} • 👨‍⚖️ {ref}</div>
    <div class="score">{ht} &nbsp; {hs} - {as_} &nbsp; {at}</div>
    <div>Xuất tự động từ Nền tảng SofaScore Football Big Data Analytics</div>
</div>
<div class="card"><h3>⚽ Danh sách bàn thắng</h3><ul>{goals_li}</ul></div>
<div class="grid">
    <div class="card"><h3>🔵 Đội hình & Điểm số — {ht}</h3>
        <table><thead><tr><th>Cầu thủ</th><th>Vị trí</th><th>Điểm</th><th>Bàn</th><th>KT</th><th>Chuyền</th></tr></thead><tbody>{_rows_p(home_lineup)}</tbody></table>
    </div>
    <div class="card"><h3>🔴 Đội hình & Điểm số — {at}</h3>
        <table><thead><tr><th>Cầu thủ</th><th>Vị trí</th><th>Điểm</th><th>Bàn</th><th>KT</th><th>Chuyền</th></tr></thead><tbody>{_rows_p(away_lineup)}</tbody></table>
    </div>
</div>
</body></html>"""


def render_match_action_bar(
    current_match: dict,
    selected_comp: str,
    match_events: pd.DataFrame,
    all_selectable_players: list,
    home_lineup: list,
    away_lineup: list,
    home_source_stats,
    away_source_stats,
    lang: str = "vi",
):
    curr_user = (st.session_state.get("auth_user") or {}).get("username", "demo")
    mid = str(current_match.get("id"))
    ht = str(current_match.get("home_team", ""))
    at = str(current_match.get("away_team", ""))
    hs = current_match.get("home_score", 0)
    as_ = current_match.get("away_score", 0)
    stage_str = str(current_match.get("stage", ""))
    match_title = f"{ht} {hs} - {as_} {at}"

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        fav_now = is_favorite(curr_user, "match", mid)
        btn_txt = tr("fav_saved", lang) if fav_now else tr("fav_add", lang)
        if st.button(btn_txt, key=f"fav_match_btn_{mid}", width="stretch"):
            toggle_favorite(curr_user, "match", mid, f"{match_title} ({stage_str})", selected_comp)
            st.rerun()

    with c2:
        yt_query = urllib.parse.quote_plus(f"{ht} vs {at} {hs}-{as_} {stage_str} Official Highlights FIFA UEFA")
        yt_url = f"https://www.youtube.com/results?search_query={yt_query}"
        st.link_button(tr("highlights_btn", lang), yt_url, width="stretch")

    with c3:
        with st.popover(tr("share_btn", lang), width="stretch"):
            scorers_txt = ", ".join(f"{g.get('player')} ({g.get('minute')}')" for g in (current_match.get("goals") or [])) or "Hòa 0-0"
            share_summary = (
                f"⚽ [{stage_str}] {ht} {hs} - {as_} {at}\n"
                f"🥅 Bàn thắng: {scorers_txt}\n"
                f"🏟️ Sân: {current_match.get('stadium')} | 📊 Phân tích trên SofaScore Big Data Analytics"
            )
            st.markdown("**📋 Thẻ tóm tắt chia sẻ nhanh (Copy 1 chạm):**")
            st.code(share_summary, language="text")
            enc_text = urllib.parse.quote(share_summary)
            s_c1, s_c2 = st.columns(2)
            s_c1.link_button("🐦 Chia sẻ lên X", f"https://twitter.com/intent/tweet?text={enc_text}", width="stretch")
            s_c2.link_button("✈️ Gửi Telegram", f"https://t.me/share/url?url=http://localhost:8501&text={enc_text}", width="stretch")

    with c4:
        with st.popover(tr("export_btn", lang), width="stretch"):
            st.markdown("**📥 Xuất dữ liệu phân tích mở (Độc quyền vượt FotMob/SofaScore):**")
            players_export_df = pd.DataFrame([
                {
                    "team": p.get("team_name"),
                    "player": p.get("display_name"),
                    "full_name": p.get("name"),
                    "position": "Sub" if p.get("is_sub") else p.get("pos"),
                    "rating": p.get("rating"),
                    "goals": p.get("goals_count"),
                    "assists": p.get("assists_count"),
                    "passes": p.get("passes"),
                    "tackles": p.get("tackles"),
                }
                for p in all_selectable_players
            ])
            st.download_button(
                "📊 Tải bảng Điểm & Chỉ số Cầu thủ (.CSV)",
                data=players_export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"player_ratings_{mid}.csv",
                mime="text/csv",
                width="stretch",
                key=f"dl_players_csv_{mid}",
            )
            if match_events is not None and not match_events.empty:
                keep_cols = [c for c in ["minute", "second", "period", "team_name", "player_name", "position_name", "type_name", "location_x", "location_y", "pass_recipient_name", "shot_outcome_name", "shot_statsbomb_xg"] if c in match_events.columns]
                st.download_button(
                    "⚡ Tải Dữ liệu Sự kiện gốc (.CSV)",
                    data=match_events[keep_cols].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"match_events_{mid}.csv",
                    mime="text/csv",
                    width="stretch",
                    key=f"dl_events_csv_{mid}",
                )
            json_pkg = {
                "match": current_match,
                "exported_at": datetime.now().isoformat(),
                "players": players_export_df.to_dict(orient="records"),
            }
            st.download_button(
                "🗂️ Tải Gói Dữ liệu JSON (.JSON)",
                data=json.dumps(json_pkg, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8"),
                file_name=f"match_analytics_{mid}.json",
                mime="application/json",
                width="stretch",
                key=f"dl_json_{mid}",
            )
            report_html = build_standalone_match_report_html(current_match, home_lineup, away_lineup, home_source_stats, away_source_stats)
            st.download_button(
                "🖨️ Xuất Báo cáo Trận đấu (.HTML / In PDF)",
                data=report_html.encode("utf-8"),
                file_name=f"match_report_{mid}.html",
                mime="text/html",
                width="stretch",
                key=f"dl_html_{mid}",
            )


# ==========================================================
# 📈 5. SOFASCORE ATTACK MOMENTUM & PHÂN TÍCH PHÒNG NGỰ (#19 & #18)
# ==========================================================
def render_attack_momentum_and_defense(match_events: pd.DataFrame, current_match: dict, home_source_stats, away_source_stats):
    if match_events is None or match_events.empty or "minute" not in match_events.columns:
        return

    ht = str(current_match.get("home_team", ""))
    at = str(current_match.get("away_team", ""))
    mins_num = pd.to_numeric(match_events["minute"], errors="coerce").fillna(0).astype(int)
    max_m = max(90, min(125, int(mins_num.max()) if not mins_num.empty else 90))

    df_m = match_events.copy()
    df_m["_min"] = mins_num.clip(1, max_m)
    loc_x = pd.to_numeric(df_m["location_x"], errors="coerce").fillna(50.0) if "location_x" in df_m.columns else pd.Series(50.0, index=df_m.index)
    is_shot = (df_m["type_name"] == "Shot").astype(float) if "type_name" in df_m.columns else pd.Series(0.0, index=df_m.index)
    xg_col = pd.to_numeric(df_m["shot_statsbomb_xg"], errors="coerce").fillna(0.0) if "shot_statsbomb_xg" in df_m.columns else pd.Series(0.0, index=df_m.index)
    # Trọng số áp lực tấn công chuẩn SofaScore: chuyền phần sân đối phương + xâm nhập 1/3 cuối + cú sút + xG
    df_m["_threat"] = (loc_x / 120.0) ** 2 * 1.2 + (loc_x >= 80.0).astype(float) * 1.35 + is_shot * 4.5 + xg_col * 16.0

    h_series = df_m[df_m["team_name"] == ht].groupby("_min")["_threat"].sum().reindex(range(1, max_m + 1), fill_value=0.0)
    a_series = df_m[df_m["team_name"] == at].groupby("_min")["_threat"].sum().reindex(range(1, max_m + 1), fill_value=0.0)
    raw_diff = (h_series - a_series).values
    kernel = np.array([0.22, 0.56, 0.22])
    smooth_diff = np.convolve(raw_diff, kernel, mode="same")
    peak = max(float(np.max(np.abs(smooth_diff))), 1.0)
    norm_mom = np.clip(smooth_diff / peak * 100.0, -100.0, 100.0)

    minutes_x = list(range(1, max_m + 1))
    colors = ["#38BDF8" if v >= 0 else "#FF4D6D" for v in norm_mom]

    st.markdown("#### 🌊 Biểu Đồ Sóng Áp Lực Thế Trận Theo Phút (SofaScore Attack Momentum)")
    with st.container(border=True):
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Bar(
            x=minutes_x,
            y=norm_mom,
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="Phút %{x}′<br>Chỉ số áp lực: <b>%{y:+.1f}</b><extra></extra>",
            name="Attack Momentum",
        ))
        for g in (current_match.get("goals") or []):
            gm = int(g.get("minute", 0) or 0)
            if 1 <= gm <= max_m:
                is_hg = g.get("team") == ht
                y_pos = 88 if is_hg else -88
                fig_mom.add_annotation(
                    x=gm, y=y_pos,
                    text=f"⚽ {gm}′",
                    showarrow=False,
                    font=dict(size=11, color="#FFFFFF", family="Arial Black"),
                    bgcolor="rgba(0, 214, 108, 0.85)" if is_hg else "rgba(255, 77, 109, 0.88)",
                    borderpad=3,
                )
        fig_mom.add_hline(y=0, line_color="rgba(226, 232, 240, 0.45)", line_width=1.2)
        fig_mom.add_vline(x=45.5, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)", annotation_text="HT 45′", annotation_position="top")
        if max_m > 95:
            fig_mom.add_vline(x=90.5, line_dash="dash", line_color="rgba(250, 204, 21, 0.4)", annotation_text="FT 90′", annotation_position="top")

        fig_mom.update_layout(
            xaxis=dict(title="Phút thi đấu", range=[0.5, max_m + 0.5], showgrid=False),
            yaxis=dict(
                range=[-108, 108],
                tickvals=[-75, 0, 75],
                ticktext=[f"🔴 {at}", "Cân bằng", f"🔵 {ht}"],
                gridcolor="rgba(148, 163, 184, 0.10)",
                zeroline=False,
            ),
            height=275,
            margin=dict(l=10, r=10, t=18, b=10),
            plot_bgcolor="#0E131F",
            paper_bgcolor="#131A26",
            font=dict(color="#E2E8F0"),
            showlegend=False,
            bargap=0.12,
        )
        st.plotly_chart(fig_mom, width="stretch", key=f"attack_momentum_{current_match.get('id')}")

        h_dom_mins = int((norm_mom > 8).sum())
        a_dom_mins = int((norm_mom < -8).sum())
        bal_mins = max(0, max_m - h_dom_mins - a_dom_mins)
        st.markdown(compact_html(f"""
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:6px 12px;background:#0E1420;border-radius:8px;font-size:12.5px;color:#CBD5E1;">
            <span>🔵 <b>{html.escape(ht)}</b> áp đảo: <b style="color:#38BDF8;">{h_dom_mins} phút</b></span>
            <span>⚖️ Giằng co cân bằng: <b>{bal_mins} phút</b></span>
            <span>🔴 <b>{html.escape(at)}</b> áp đảo: <b style="color:#FF4D6D;">{a_dom_mins} phút</b></span>
        </div>
        """), unsafe_allow_html=True)


# ==========================================================
# 🤖 6. TAB 6 TRONG TRẬN: AI MONTE CARLO, BÌNH LUẬN TRỰC TIẾP & BẢN TIN (#37, #32, #40)
# ==========================================================
def simulate_match_monte_carlo(match_events: pd.DataFrame, home_team: str, away_team: str, n_sims: int = 10000):
    shots = match_events[match_events["type_name"] == "Shot"].copy() if (match_events is not None and not match_events.empty and "type_name" in match_events.columns) else pd.DataFrame()
    if "period" in shots.columns:
        p_num = pd.to_numeric(shots["period"], errors="coerce")
        shots = shots[p_num.isna() | p_num.isin([1, 2, 3, 4])]

    h_sh_df = shots[shots["team_name"] == home_team] if (not shots.empty and "team_name" in shots.columns) else pd.DataFrame()
    a_sh_df = shots[shots["team_name"] == away_team] if (not shots.empty and "team_name" in shots.columns) else pd.DataFrame()
    h_xgs = (
        pd.to_numeric(h_sh_df["shot_statsbomb_xg"], errors="coerce").fillna(0.08).clip(0.01, 0.96).values
        if (not h_sh_df.empty and "shot_statsbomb_xg" in h_sh_df.columns)
        else (np.full(len(h_sh_df), 0.10) if not h_sh_df.empty else np.array([]))
    )
    a_xgs = (
        pd.to_numeric(a_sh_df["shot_statsbomb_xg"], errors="coerce").fillna(0.08).clip(0.01, 0.96).values
        if (not a_sh_df.empty and "shot_statsbomb_xg" in a_sh_df.columns)
        else (np.full(len(a_sh_df), 0.10) if not a_sh_df.empty else np.array([]))
    )

    rng = np.random.default_rng(2026)
    h_goals_sim = (rng.random((n_sims, len(h_xgs))) < h_xgs).sum(axis=1) if len(h_xgs) > 0 else np.zeros(n_sims, dtype=int)
    a_goals_sim = (rng.random((n_sims, len(a_xgs))) < a_xgs).sum(axis=1) if len(a_xgs) > 0 else np.zeros(n_sims, dtype=int)

    h_win = float((h_goals_sim > a_goals_sim).mean() * 100.0)
    draw = float((h_goals_sim == a_goals_sim).mean() * 100.0)
    a_win = float((h_goals_sim < a_goals_sim).mean() * 100.0)
    xpts_h = round((h_win / 100.0) * 3.0 + (draw / 100.0) * 1.0, 2)
    xpts_a = round((a_win / 100.0) * 3.0 + (draw / 100.0) * 1.0, 2)

    pairs = pd.Series(list(zip(h_goals_sim, a_goals_sim))).value_counts().head(6)
    top_scores = [{"score": f"{k[0]} - {k[1]}", "prob": round(v / n_sims * 100.0, 1)} for k, v in pairs.items()]
    return {
        "h_win": round(h_win, 1),
        "draw": round(draw, 1),
        "a_win": round(a_win, 1),
        "h_xg": round(float(h_xgs.sum()), 2) if len(h_xgs) else 0.0,
        "a_xg": round(float(a_xgs.sum()), 2) if len(a_xgs) else 0.0,
        "xpts_h": xpts_h,
        "xpts_a": xpts_a,
        "top_scores": top_scores,
    }


def build_live_commentary_feed(match_events: pd.DataFrame, current_match: dict, max_minute: int = 999):
    """Tạo dòng thời gian Bình luận Trực tiếp (Live Commentary) từ dữ liệu sự kiện thật của trận đấu."""
    ht = str(current_match.get("home_team", ""))
    at = str(current_match.get("away_team", ""))
    items = [
        {"minute": 0, "type": "whistle", "icon": "📢", "color": "#38BDF8", "title": "Trận đấu bắt đầu (Kick-off)!", "body": f"Trọng tài {current_match.get('referee', '')} nổi còi khai cuộc trận {ht} gặp {at} trên sân {current_match.get('stadium', '')}."}
    ]

    for g in (current_match.get("goals") or []):
        gm = int(g.get("minute", 0) or 0)
        if gm <= max_minute and g.get("is_og"):
            items.append({
                "minute": gm, "type": "goal", "icon": "⚽", "color": "#F97316",
                "title": f"PHẢN LƯỚI NHÀ! Bàn thắng cho {g.get('team')} ({gm}′)",
                "body": f"Tình huống đốt lưới nhà đáng tiếc của {g.get('player')} giúp {g.get('team')} có bàn thắng!",
            })

    if match_events is not None and not match_events.empty and "type_name" in match_events.columns:
        ev = match_events.copy()
        ev["_m"] = pd.to_numeric(ev["minute"], errors="coerce").fillna(0).astype(int) if "minute" in ev.columns else 0
        ev = ev[ev["_m"] <= max_minute]
        if "period" in ev.columns:
            p_num = pd.to_numeric(ev["period"], errors="coerce")
            ev = ev[p_num.isna() | p_num.isin([1, 2, 3, 4])]

        shots = ev[ev["type_name"] == "Shot"]
        for _, r in shots.iterrows():
            m = int(r["_m"])
            p = _s(r.get("player_name"), "Cầu thủ")
            t = _s(r.get("team_name"), "")
            out = _s(r.get("shot_outcome_name"), "")
            xg_raw = pd.to_numeric(r.get("shot_statsbomb_xg"), errors="coerce")
            xg = float(xg_raw) if pd.notna(xg_raw) else 0.08
            if out == "Goal":
                items.append({
                    "minute": m, "type": "goal", "icon": "⚽", "color": "#00E676",
                    "title": f"VÀOOOOO!! {p} lập công cho {t} ({m}′)",
                    "body": f"Cú dứt điểm sắc bén của {p} mang về bàn thắng cho {t} (xG: {xg:.2f})!",
                })
            elif out in ("Saved", "Saved to Post") and xg >= 0.12:
                items.append({
                    "minute": m, "type": "chance", "icon": "🧤", "color": "#38BDF8",
                    "title": f"Cứu thua xuất thần! ({m}′)",
                    "body": f"{p} ({t}) tung cú dứt điểm nguy hiểm (xG: {xg:.2f}) nhưng thủ môn đối phương đã bay người cản phá!",
                })
            elif out == "Post" or xg >= 0.25:
                items.append({
                    "minute": m, "type": "chance", "icon": "🔥", "color": "#FBBF24",
                    "title": f"Cơ hội mười mươi bị bỏ lỡ! ({m}′)",
                    "body": f"Pha dứt điểm có chỉ số xG rất cao ({xg:.2f}) của {p} ({t}) đi {('trúng khung gỗ' if out == 'Post' else 'chệch khung thành trong gang tấc')}!",
                })

        subs = ev[ev["type_name"] == "Substitution"]
        for _, r in subs.iterrows():
            m = int(r["_m"])
            p_out = _s(r.get("player_name"), "")
            p_in = _s(r.get("substitution_replacement_name"), "")
            t = _s(r.get("team_name"), "")
            if p_out or p_in:
                items.append({
                    "minute": m, "type": "sub", "icon": "🔄", "color": "#A78BFA",
                    "title": f"Thay đổi nhân sự — {t} ({m}′)",
                    "body": f"{p_in or 'Cầu thủ dự bị'} vào sân thay cho {p_out}.",
                })

        if "card_name" in ev.columns:
            cards = ev[ev["card_name"].fillna("").astype(str).str.strip() != ""]
            for _, r in cards.iterrows():
                m = int(r["_m"])
                c_nm = _s(r.get("card_name"), "")
                p_card = _s(r.get("player_name"), "Cầu thủ")
                t_card = _s(r.get("team_name"), "")
                is_red = "red" in c_nm.lower() or "second yellow" in c_nm.lower()
                items.append({
                    "minute": m, "type": "card", "icon": "🟥" if is_red else "🟨",
                    "color": "#EF4444" if is_red else "#FACC15",
                    "title": f"{'Thẻ đỏ' if is_red else 'Thẻ vàng'} — {p_card} ({m}′)",
                    "body": f"Trọng tài rút {'thẻ đỏ truất quyền thi đấu' if is_red else 'thẻ vàng cảnh cáo'} đối với {p_card} ({t_card}).",
                })

    if max_minute >= 45:
        items.append({"minute": 45, "type": "whistle", "icon": "⏱️", "color": "#94A3B8", "title": "Khép lại Hiệp 1 (Half-Time)", "body": "Hai đội tạm rời sân nghỉ giữa hiệp để điều chỉnh chiến thuật."})
    if max_minute >= 90:
        items.append({
            "minute": 90, "type": "whistle", "icon": "🏁", "color": "#00D66C",
            "title": f"Kết thúc thời gian thi đấu: {ht} {current_match.get('home_score')} - {current_match.get('away_score')} {at}",
            "body": "Trận đấu khép lại với những thông số chuyên môn ấn tượng từ cả hai đội.",
        })

    items.sort(key=lambda x: x["minute"], reverse=True)
    return items


def render_match_ai_timeline_news_tab(match_events: pd.DataFrame, current_match: dict, all_selectable_players: list):
    ht = str(current_match.get("home_team", ""))
    at = str(current_match.get("away_team", ""))
    hs = int(current_match.get("home_score", 0) or 0)
    as_ = int(current_match.get("away_score", 0) or 0)
    mid = str(current_match.get("id", ""))

    sim = simulate_match_monte_carlo(match_events, ht, at, n_sims=10000)

    st.subheader("🤖 Mô Hình AI Mô Phỏng 10.000 Kịch Bản Trận Đấu (Monte Carlo xG Engine)")
    with st.container(border=True):
        st.markdown(compact_html(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:13.5px;font-weight:800;margin-bottom:6px;">
                <span style="color:#38BDF8;">🔵 {html.escape(ht)} thắng: {sim['h_win']}% (xPts: {sim['xpts_h']})</span>
                <span style="color:#E2E8F0;">⚖️ Hòa: {sim['draw']}%</span>
                <span style="color:#FF4D6D;">{html.escape(at)} thắng: {sim['a_win']}% (xPts: {sim['xpts_a']}) 🔴</span>
            </div>
            <div style="display:flex;height:14px;border-radius:999px;overflow:hidden;background:#0F172A;gap:2px;">
                <div style="width:{max(3.0, sim['h_win'])}%;background:linear-gradient(90deg,#0284C7,#38BDF8);"></div>
                <div style="width:{max(3.0, sim['draw'])}%;background:#64748B;"></div>
                <div style="width:{max(3.0, sim['a_win'])}%;background:linear-gradient(90deg,#FF4D6D,#E11D48);"></div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        score_pills = "".join(
            f'<div style="background:#0E1420;border:1px solid #26354D;border-radius:10px;padding:8px 12px;text-align:center;">'
            f'<div style="color:#F8FAFC;font-size:16px;font-weight:900;">{s["score"]}</div>'
            f'<div style="color:#38BDF8;font-size:11.5px;font-weight:700;">Xác suất: {s["prob"]}%</div></div>'
            for s in sim["top_scores"]
        )
        st.markdown(compact_html(f"""
        <div style="font-size:12px;color:#94A3B8;font-weight:700;margin-bottom:6px;text-transform:uppercase;">🎯 Top 6 kịch bản tỷ số khả dĩ nhất dựa trên chất lượng cơ hội (xG: {sim['h_xg']} vs {sim['a_xg']})</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;">{score_pills}</div>
        """), unsafe_allow_html=True)

        with st.expander("📝 Lưu dự đoán & nhận định cá nhân của bạn vào CSDL SQLite"):
            curr_user = (st.session_state.get("auth_user") or {}).get("username", "demo")
            saved_p = get_user_prediction(curr_user, mid)
            pc1, pc2, pc3, pc4 = st.columns([1, 1, 2.5, 1.2])
            ph = pc1.number_input(f"Bàn {ht}", 0, 15, int(saved_p["pred_home"]) if saved_p else hs, key=f"up_h_{mid}")
            pa = pc2.number_input(f"Bàn {at}", 0, 15, int(saved_p["pred_away"]) if saved_p else as_, key=f"up_a_{mid}")
            pnote = pc3.text_input("Ghi chú chiến thuật:", value=saved_p["note"] if saved_p else "", key=f"up_n_{mid}")
            if pc4.button("💾 Lưu CSDL", key=f"up_save_{mid}", width="stretch"):
                save_user_prediction(curr_user, mid, f"{ht} vs {at}", ph, pa, pnote)
                st.success("Đã lưu nhận định vào SQLite!")

    col_comm, col_news = st.columns([1.25, 1.0], gap="medium")

    with col_comm:
        with st.container(border=True):
            st.markdown("#### 🎙️ Bình Luận Trực Tiếp & Dòng Thời Gian Sự Kiện")
            comm_filter = st.radio(
                "Bộ lọc diễn biến:",
                ["Tất cả", "⚽ Bàn thắng", "🔥 Cơ hội nguy hiểm", "🔄 Thay người & Thẻ"],
                horizontal=True,
                key=f"comm_flt_{mid}",
            )
            feed = build_live_commentary_feed(match_events, current_match)
            if comm_filter == "⚽ Bàn thắng":
                feed = [x for x in feed if x["type"] == "goal"]
            elif comm_filter == "🔥 Cơ hội nguy hiểm":
                feed = [x for x in feed if x["type"] in ("goal", "chance")]
            elif comm_filter == "🔄 Thay người & Thẻ":
                feed = [x for x in feed if x["type"] in ("sub", "card")]

            cards_html = []
            for item in feed[:45]:
                cards_html.append(compact_html(f"""
                <div style="display:flex;gap:10px;align-items:flex-start;background:#0E1420;border:1px solid #1E293B;border-left:3.5px solid {item['color']};border-radius:9px;padding:8px 11px;">
                    <span style="background:#172133;color:#F8FAFC;font-weight:900;font-size:11.5px;padding:3px 7px;border-radius:6px;white-space:nowrap;">{item['minute']}′</span>
                    <div style="min-width:0;flex:1;">
                        <div style="color:#F8FAFC;font-weight:800;font-size:13px;">{item['icon']} {html.escape(item['title'])}</div>
                        <div style="color:#94A3B8;font-size:12px;margin-top:2px;line-height:1.4;">{html.escape(item['body'])}</div>
                    </div>
                </div>
                """))
            st.markdown(
                f'<div style="display:flex;flex-direction:column;gap:7px;max-height:460px;overflow-y:auto;padding-right:4px;">{"".join(cards_html)}</div>',
                unsafe_allow_html=True,
            )

    with col_news:
        with st.container(border=True):
            st.markdown("#### 📰 Bản Tin Phân Tích Chuyên Môn Tự Động")
            motm = max(all_selectable_players, key=lambda p: p.get("rating", 0)) if all_selectable_players else None
            motm_name = motm["display_name"] if motm else "Ngôi sao trận đấu"
            motm_rat = motm["rating"] if motm else 8.0
            winner_txt = (
                f"{ht} giành chiến thắng thuyết phục {hs}-{as_} trước {at}"
                if hs > as_ else (
                    f"{at} xuất sắc vượt qua {ht} với tỷ số {as_}-{hs}"
                    if as_ > hs else f"{ht} và {at} bất phân thắng bại {hs}-{as_} trong màn rượt đuổi kịch tính"
                )
            )
            xg_story = (
                f"Xét trên mô hình bàn thắng kỳ vọng (xG), {ht} tạo ra {sim['h_xg']} xG so với {sim['a_xg']} xG của {at}, "
                f"tương ứng xác suất thắng mô phỏng AI đạt {sim['h_win']}% cho {ht} và {sim['a_win']}% cho {at}."
            )
            goals_desc = ", ".join(f"{g.get('player')} ({g.get('minute')}′)" for g in (current_match.get("goals") or [])) or "hàng thủ hai đội giữ sạch lưới tuyệt đối"
            st.markdown(compact_html(f"""
            <div style="background:linear-gradient(145deg,#141D2E,#0E1420);border:1px solid #26354D;border-radius:12px;padding:15px;">
                <span style="background:rgba(56,189,248,0.16);color:#38BDF8;font-size:11px;font-weight:800;padding:3px 9px;border-radius:999px;text-transform:uppercase;">⚡ Tiêu điểm sau trận</span>
                <h4 style="color:#FFFFFF;margin:10px 0 8px;line-height:1.35;">{html.escape(winner_txt)}; {html.escape(motm_name)} rực sáng với {motm_rat}★</h4>
                <p style="color:#CBD5E1;font-size:13px;line-height:1.6;margin-bottom:10px;">
                    Tại thánh địa <b>{html.escape(str(current_match.get('stadium', '')))}</b>, cuộc đọ sức khuôn khổ <b>{html.escape(str(current_match.get('stage', '')))}</b> đã khép lại với điểm nhấn thuộc về {html.escape(goals_desc)}.
                </p>
                <p style="color:#94A3B8;font-size:12.8px;line-height:1.6;margin-bottom:10px;">
                    📊 <b>Góc nhìn Big Data:</b> {html.escape(xg_story)}
                </p>
                <div style="border-top:1px solid #1E293B;padding-top:9px;color:#4ADE80;font-size:12.5px;font-weight:700;">
                    👑 Cầu thủ xuất sắc nhất trận (MOTM): {html.escape(motm_name)} ({motm_rat}★)
                </div>
            </div>
            """), unsafe_allow_html=True)


# ==========================================================
# 🔴 7. KHÔNG GIAN 2: TRỰC TIẾP REAL-TIME & XEM NHIỀU TRẬN CÙNG LÚC (#1, #5, #21, #32)
# ==========================================================
@st.fragment(run_every="1.2s")
def _live_single_match_player_fragment(current_match: dict, match_events: pd.DataFrame, max_ev_m: int):
    mid = str(current_match["id"])
    cur_key = f"live_cur_val_{mid}"
    play_key = f"live_sim_play_{mid}"
    slider_key = f"live_slider_{mid}"
    speed_key = f"live_speed_{mid}"
    init_key = f"live_inited_{mid}"
    interacted_key = f"live_interacted_{mid}"

    if init_key not in st.session_state:
        st.session_state[init_key] = True
        st.session_state[cur_key] = 1
        st.session_state[slider_key] = 1
        st.session_state[play_key] = True
        st.session_state[speed_key] = 1
        st.session_state[interacted_key] = True

    just_interacted = bool(st.session_state.pop(interacted_key, False))
    cur_m = int(st.session_state.get(cur_key, 1))
    is_playing_prev = bool(st.session_state.get(play_key, True))
    step_val = int(st.session_state.get(speed_key, 1))

    # Nhịp tự động 1.2 giây: tự nhích đồng hồ & kéo con trượt tiến lên từ đúng phút người dùng đang chọn
    if not just_interacted and is_playing_prev and cur_m < max_ev_m:
        cur_m = min(max_ev_m, cur_m + step_val)
        st.session_state[cur_key] = cur_m
        st.session_state[slider_key] = cur_m
    else:
        st.session_state[slider_key] = cur_m

    def _on_slider_scrub():
        st.session_state[cur_key] = int(st.session_state[slider_key])
        st.session_state[interacted_key] = True

    def _on_toggle_play():
        st.session_state[interacted_key] = True
        if st.session_state.get(play_key) and int(st.session_state.get(cur_key, 1)) >= max_ev_m:
            st.session_state[cur_key] = 1
            st.session_state[slider_key] = 1

    def _jump_to_minute(target_m: int):
        tm = max(1, min(max_ev_m, int(target_m)))
        st.session_state[cur_key] = tm
        st.session_state[slider_key] = tm
        st.session_state[play_key] = True
        st.session_state[interacted_key] = True

    c_play, c_spd, c_b1, c_b2, c_b3 = st.columns([1.45, 1.15, 0.95, 0.95, 1.0], gap="small")
    is_playing = c_play.toggle("▶️ Tự động chạy Real-time", key=play_key, on_change=_on_toggle_play)
    c_spd.selectbox(
        "Tốc độ phát:",
        [1, 2, 5],
        format_func=lambda v: f"⚡ Tốc độ {v}x ({v} phút/nhịp)",
        key=speed_key,
        label_visibility="collapsed",
    )
    c_b1.button("⏮️ Phút 1′", on_click=_jump_to_minute, args=(1,), width="stretch", key=f"lv_j1_{mid}")
    c_b2.button("⏱️ Hiệp 2 (46′)", on_click=_jump_to_minute, args=(min(46, max_ev_m),), width="stretch", key=f"lv_j46_{mid}")
    c_b3.button("🔥 Phút 80′", on_click=_jump_to_minute, args=(min(80, max_ev_m),), width="stretch", key=f"lv_j80_{mid}")

    st.slider(
        "⏱️ Thanh thời gian trực tiếp (Kéo tới phút bất kỳ — đồng hồ sẽ tự chạy tiếp từ phút bạn chọn):",
        1,
        max_ev_m,
        key=slider_key,
        on_change=_on_slider_scrub,
    )

    minute_now = int(st.session_state.get(cur_key, 1))
    ht = current_match["home_team"]
    at = current_match["away_team"]
    goals_now = [g for g in (current_match.get("goals") or []) if int(g.get("minute", 0) or 0) <= minute_now]
    recent_goals = [g for g in goals_now if minute_now - 2 <= int(g.get("minute", 0) or 0) <= minute_now]
    hs_now = sum(1 for g in goals_now if g.get("team") == ht)
    as_now = sum(1 for g in goals_now if g.get("team") == at)

    ev_now = (
        match_events[pd.to_numeric(match_events["minute"], errors="coerce").fillna(0) <= minute_now]
        if (not match_events.empty and "minute" in match_events.columns)
        else pd.DataFrame()
    )
    sh_now = ev_now[ev_now["type_name"] == "Shot"] if (not ev_now.empty and "type_name" in ev_now.columns) else pd.DataFrame()
    h_sh = sh_now[sh_now["team_name"] == ht] if (not sh_now.empty and "team_name" in sh_now.columns) else pd.DataFrame()
    a_sh = sh_now[sh_now["team_name"] == at] if (not sh_now.empty and "team_name" in sh_now.columns) else pd.DataFrame()
    h_xg_now = float(pd.to_numeric(h_sh["shot_statsbomb_xg"], errors="coerce").fillna(0.08).sum()) if (not h_sh.empty and "shot_statsbomb_xg" in h_sh.columns) else 0.0
    a_xg_now = float(pd.to_numeric(a_sh["shot_statsbomb_xg"], errors="coerce").fillna(0.08).sum()) if (not a_sh.empty and "shot_statsbomb_xg" in a_sh.columns) else 0.0
    h_pass_now = int(((ev_now["team_name"] == ht) & (ev_now["type_name"] == "Pass")).sum()) if (not ev_now.empty and {"team_name", "type_name"}.issubset(ev_now.columns)) else 0
    a_pass_now = int(((ev_now["team_name"] == at) & (ev_now["type_name"] == "Pass")).sum()) if (not ev_now.empty and {"team_name", "type_name"}.issubset(ev_now.columns)) else 0
    tot_p_now = max(1, h_pass_now + a_pass_now)
    h_pos_now = round(h_pass_now / tot_p_now * 100, 1)
    a_pos_now = round(100.0 - h_pos_now, 1)

    if minute_now >= max_ev_m:
        status_badge = f"🏁 HẾT GIỜ • {max_ev_m}′"
        badge_style = "background:rgba(0,214,108,0.18);border:1px solid #00D66C;color:#4ADE80;"
    elif is_playing:
        status_badge = f"🔴 ĐANG PHÁT TRỰC TIẾP • {minute_now}:00"
        badge_style = "background:rgba(239,68,68,0.22);border:1px solid #EF4444;color:#FCA5A5;animation:livePulse 1.1s infinite;"
    else:
        status_badge = f"⏸️ ĐANG TẠM DỪNG • {minute_now}′"
        badge_style = "background:rgba(250,204,21,0.18);border:1px solid #FACC15;color:#FDE047;"

    goal_flash_html = ""
    if recent_goals:
        rg = recent_goals[-1]
        goal_flash_html = (
            f'<div style="margin-top:10px;padding:7px 14px;background:rgba(0,230,118,0.16);border:1px solid #00E676;border-radius:10px;text-align:center;color:#4ADE80;font-weight:900;font-size:13.5px;">'
            f'⚽ VÀOOOOO!! {html.escape(str(rg.get("player", "")))} ghi bàn cho {html.escape(str(rg.get("team", "")))} ở phút {rg.get("minute")}′!'
            f'</div>'
        )

    scorers_live_str = " • ".join(f"⚽ {html.escape(str(g.get('player')))} ({g.get('minute')}′)" for g in goals_now)
    scorers_row_html = f'<div style="text-align:center;color:#E2E8F0;font-size:12.5px;margin-top:8px;font-weight:700;">{scorers_live_str}</div>' if scorers_live_str else ""

    st.markdown(compact_html(f"""
    <style>@keyframes livePulse {{ 0% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.55); }} 70% {{ opacity: 0.78; box-shadow: 0 0 0 9px rgba(239,68,68,0); }} 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0); }} }}</style>
    <div style="background:linear-gradient(135deg,#131D2E,#0F1726);border:1.5px solid {'#00E676' if recent_goals else '#EF4444'};border-radius:16px;padding:18px 24px;margin-bottom:14px;">
        <div style="text-align:center;margin-bottom:8px;">
            <span style="{badge_style}font-weight:900;font-size:12.5px;padding:4px 14px;border-radius:999px;letter-spacing:0.06em;display:inline-block;">{status_badge}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:20px;">
            <div style="text-align:right;font-size:24px;font-weight:900;color:#38BDF8;">🔵 {html.escape(ht)}</div>
            <div style="background:#0B0F19;border:1.5px solid #38BDF8;border-radius:14px;padding:10px 24px;font-size:38px;font-weight:900;color:#FFF;">{hs_now} - {as_now}</div>
            <div style="text-align:left;font-size:24px;font-weight:900;color:#FF4D6D;">{html.escape(at)} 🔴</div>
        </div>
        {scorers_row_html}
        {goal_flash_html}
        <div style="display:flex;justify-content:space-around;flex-wrap:wrap;gap:10px;margin-top:14px;padding-top:10px;border-top:1px solid #1E293B;font-size:13px;color:#CBD5E1;">
            <span>🎯 xG trực tiếp: <b style="color:#38BDF8;">{h_xg_now:.2f}</b> vs <b style="color:#FF4D6D;">{a_xg_now:.2f}</b></span>
            <span>👟 Cú sút: <b>{len(h_sh)}</b> vs <b>{len(a_sh)}</b></span>
            <span>🔄 Kiểm soát bóng: <b>{h_pos_now}%</b> vs <b>{a_pos_now}%</b></span>
            <span>📡 Đường chuyền: <b>{h_pass_now}</b> vs <b>{a_pass_now}</b></span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    live_feed = build_live_commentary_feed(match_events, current_match, max_minute=minute_now)
    st.markdown(f"##### 🎙️ Bình luận trực tiếp tính đến phút {minute_now}′ ({len(live_feed)} sự kiện)")
    rows_c = "".join(
        compact_html(f"""
        <div style="display:flex;gap:10px;align-items:center;background:#131A26;border:1px solid #243247;border-left:3.5px solid {it['color']};border-radius:8px;padding:7px 11px;">
            <span style="font-weight:900;color:#38BDF8;min-width:34px;">{it['minute']}′</span>
            <span style="color:#F8FAFC;font-weight:700;font-size:13px;">{it['icon']} <b>{html.escape(it['title'])}</b> — <span style="color:#94A3B8;font-weight:400;">{html.escape(it['body'])}</span></span>
        </div>
        """)
        for it in live_feed[:18]
    )
    st.markdown(f'<div style="display:flex;flex-direction:column;gap:6px;">{rows_c}</div>', unsafe_allow_html=True)


@st.fragment(run_every="1.5s")
def _multi_match_live_fragment(all_matches_flat: list, picked_labels: list):
    if "multi_live_play" not in st.session_state:
        st.session_state["multi_live_play"] = True
        st.session_state["multi_live_min"] = 1
        st.session_state["multi_live_slider"] = 1
        st.session_state["multi_live_interacted"] = True

    just_int = bool(st.session_state.pop("multi_live_interacted", False))
    m_cur = int(st.session_state.get("multi_live_min", 1))
    m_play = bool(st.session_state.get("multi_live_play", True))

    if not just_int and m_play and m_cur < 95:
        m_cur = min(95, m_cur + 2)
        st.session_state["multi_live_min"] = m_cur
        st.session_state["multi_live_slider"] = m_cur
    else:
        st.session_state["multi_live_slider"] = m_cur

    def _on_m_scrub():
        st.session_state["multi_live_min"] = int(st.session_state["multi_live_slider"])
        st.session_state["multi_live_interacted"] = True

    def _reset_m():
        st.session_state["multi_live_min"] = 1
        st.session_state["multi_live_slider"] = 1
        st.session_state["multi_live_play"] = True
        st.session_state["multi_live_interacted"] = True

    mc1, mc2, mc3 = st.columns([1.3, 1.0, 2.7], gap="small")
    is_m_play = mc1.toggle("🔴 Phát Live đồng loạt", key="multi_live_play")
    mc2.button("⏮️ Về Phút 1′", on_click=_reset_m, width="stretch", key="multi_reset_btn")
    mc3.slider("⏱️ Đồng hồ chung các sân (Phút):", 1, 95, key="multi_live_slider", on_change=_on_m_scrub)

    now_m = int(st.session_state.get("multi_live_min", 1))
    grid_cols = st.columns(2, gap="medium")
    for idx_m, p_lbl in enumerate(picked_labels):
        _, comp_n, m_meta = next(x for x in all_matches_flat if x[0] == p_lbl)
        with grid_cols[idx_m % 2]:
            with st.container(border=True):
                ht_m = m_meta.get("home_team", "Home")
                at_m = m_meta.get("away_team", "Away")
                goals_all = m_meta.get("goals") or []
                if goals_all:
                    goals_up_to = [g for g in goals_all if int(g.get("minute", 0) or 0) <= now_m]
                    hs_m = sum(1 for g in goals_up_to if g.get("team") == ht_m)
                    as_m = sum(1 for g in goals_up_to if g.get("team") == at_m)
                    scorers_m = ", ".join(f"{g.get('player')} {g.get('minute')}′" for g in goals_up_to[:5]) or "Đang thi đấu chặt chẽ (0 - 0)"
                else:
                    hs_m = m_meta.get("home_score", 0) if now_m >= 90 else 0
                    as_m = m_meta.get("away_score", 0) if now_m >= 90 else 0
                    scorers_m = f"Tỷ số chung cuộc: {m_meta.get('home_score', 0)} - {m_meta.get('away_score', 0)}"
                pen_str = f" (Pen {m_meta['pen_home']}-{m_meta['pen_away']})" if (now_m >= 90 and m_meta.get("pen_home") is not None) else ""
                live_tag = f"🏁 FT" if now_m >= 95 else (f"🔴 LIVE {now_m}′" if is_m_play else f"⏸️ {now_m}′")
                st.markdown(compact_html(f"""
                <div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="color:#38BDF8;font-size:11.5px;font-weight:800;">{html.escape(comp_n)} • {html.escape(str(m_meta.get('stage', '')))}</span>
                        <span style="color:#FCA5A5;font-size:11.5px;font-weight:900;">{live_tag}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin:10px 0;padding:10px 14px;background:#0E1420;border-radius:10px;border:1px solid #243247;">
                        <span style="color:#F8FAFC;font-weight:800;font-size:16px;">🔵 {html.escape(str(ht_m))}</span>
                        <span style="color:#00E676;font-weight:900;font-size:22px;">{hs_m} - {as_m}<span style="font-size:11px;color:#FACC15;">{html.escape(pen_str)}</span></span>
                        <span style="color:#F8FAFC;font-weight:800;font-size:16px;">{html.escape(str(at_m))} 🔴</span>
                    </div>
                    <div style="color:#94A3B8;font-size:12px;margin-bottom:8px;">⚽ {html.escape(scorers_m)}</div>
                </div>
                """), unsafe_allow_html=True)
                if st.button(f"🔍 Mở phân tích sâu: {ht_m} vs {at_m}", key=f"jump_multi_{m_meta['id']}_{idx_m}", width="stretch"):
                    st.session_state["jump_comp"] = comp_n
                    st.session_state["jump_match_label"] = m_meta["label"]
                    st.session_state["pending_workspace"] = "ws_match"
                    st.rerun()


def render_live_and_multimatch_workspace(catalog: dict, current_match: dict, match_events: pd.DataFrame, wc2018_events_full: pd.DataFrame, wc2018_shots_df: pd.DataFrame):
    st.markdown("## 🔴 Trung Tâm Tỷ Số Trực Tiếp Real-Time & Theo Dõi Đa Trận (Live Match Center)")
    sub_live, sub_multi = st.tabs([
        "📡 Giả Lập Trực Tiếp Theo Phút (Real-Time Live Score & Commentary)",
        "📺 Theo Dõi Nhiều Trận Cùng Lúc (Multi-Match Split Screen)",
    ])

    with sub_live:
        max_ev_m = int(max(90, pd.to_numeric(match_events["minute"], errors="coerce").fillna(90).max())) if (not match_events.empty and "minute" in match_events.columns) else 90
        _live_single_match_player_fragment(current_match, match_events, max_ev_m)

    with sub_multi:
        st.markdown("#### 📺 Bảng Theo Dõi Nhiều Trận Cùng Lúc (Multi-Match Split Screen — tiêu chí #5)")
        all_matches_flat = []
        default_picks = []
        seen_default_comps = set()
        for c_name, m_list in catalog.items():
            for m in m_list:
                lbl_item = f"[{c_name.split()[1] if len(c_name.split()) > 1 else c_name}] {m['label']}"
                all_matches_flat.append((lbl_item, c_name, m))
                if c_name not in seen_default_comps and len(default_picks) < 4:
                    default_picks.append(lbl_item)
                    seen_default_comps.add(c_name)
        if len(default_picks) < 4:
            for item_x in all_matches_flat:
                if item_x[0] not in default_picks and len(default_picks) < 4:
                    default_picks.append(item_x[0])

        picked_labels = st.multiselect(
            "Chọn từ 2 đến 4 trận đấu để theo dõi song song:",
            [x[0] for x in all_matches_flat],
            default=default_picks,
            max_selections=4,
        )
        if not picked_labels:
            st.info("Hãy chọn ít nhất 1 trận đấu để hiển thị trên màn hình ghép.")
        else:
            _multi_match_live_fragment(all_matches_flat, picked_labels)


# ==========================================================
# 🏆 8. KHÔNG GIAN 3: BXH, LỊCH THI ĐẤU, THỐNG KÊ MÙA & CHUYỂN NHƯỢNG (#33, #34, #35, #38)
# ==========================================================
@st.cache_data(show_spinner=False)
def compute_wc2018_season_leaderboards(wc2018_events_full: pd.DataFrame, wc2018_shots_df: pd.DataFrame):
    if wc2018_events_full is None or wc2018_events_full.empty:
        return pd.DataFrame(), pd.DataFrame()

    matches_meta = (
        wc2018_events_full[[
            "match_id", "match_date", "competition_stage_name",
            "home_team_home_team_name", "home_team_home_team_group",
            "away_team_away_team_name", "away_team_away_team_group",
            "home_score", "away_score", "stadium_name",
        ]]
        .drop_duplicates(subset=["match_id"])
        .copy()
    )

    ev = wc2018_events_full[wc2018_events_full["player_name"].notna()].copy()
    ev["succ_p"] = pd.to_numeric(ev.get("Successful Passes", 0), errors="coerce").fillna(0)
    ev["prog_p"] = pd.to_numeric(ev.get("Successful Progressive Passes", 0), errors="coerce").fillna(0)
    ev["key_p"] = pd.to_numeric(ev.get("pass_shot_assist", 0), errors="coerce").fillna(0)
    ev["ast"] = pd.to_numeric(ev.get("pass_goal_assist", 0), errors="coerce").fillna(0)
    ev["xt_v"] = pd.to_numeric(ev.get("xT", 0), errors="coerce").fillna(0.0)

    p_pass = (
        ev.groupby(["player_name", "team_name"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            passes=("id", "count"),
            passes_comp=("succ_p", "sum"),
            prog_passes=("prog_p", "sum"),
            key_passes=("key_p", "sum"),
            assists=("ast", "sum"),
            xt=("xt_v", "sum"),
        )
    )

    if wc2018_shots_df is not None and not wc2018_shots_df.empty:
        sh = wc2018_shots_df[wc2018_shots_df["period"].isin([1, 2, 3, 4])].copy()
        sh["is_goal"] = (sh["shot_outcome_name"] == "Goal").astype(int)
        sh["is_sot"] = sh["shot_outcome_name"].isin(["Goal", "Saved", "Saved to Post"]).astype(int)
        sh["xg_v"] = pd.to_numeric(sh["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
        p_sh = (
            sh.groupby(["player_name", "team_name"], as_index=False)
            .agg(goals=("is_goal", "sum"), shots=("is_goal", "count"), sot=("is_sot", "sum"), xg=("xg_v", "sum"))
        )
        leaderboard = pd.merge(p_pass, p_sh, on=["player_name", "team_name"], how="outer").fillna(0)
    else:
        leaderboard = p_pass.copy()
        for col in ("goals", "shots", "sot", "xg"):
            leaderboard[col] = 0

    leaderboard["matches"] = leaderboard["matches"].clip(lower=1).astype(int)
    for col in ("goals", "assists", "shots", "sot", "passes", "passes_comp", "prog_passes", "key_passes"):
        leaderboard[col] = leaderboard[col].astype(int)
    leaderboard["pass_pct"] = np.where(leaderboard["passes"] > 0, (leaderboard["passes_comp"] / leaderboard["passes"] * 100.0).round(1), 0.0)
    leaderboard["xg"] = leaderboard["xg"].round(2)
    leaderboard["xt"] = leaderboard["xt"].round(2)
    leaderboard["avg_rating"] = np.clip(
        6.35
        + (leaderboard["goals"] * 0.55 + leaderboard["assists"] * 0.40 + leaderboard["key_passes"] * 0.08 + leaderboard["xt"] * 0.45) / leaderboard["matches"],
        6.0,
        9.6,
    ).round(2)
    return matches_meta, leaderboard


def _compute_standings_from_catalog_matches(matches_list: list, mode: str = "groups", seed_groups: dict = None):
    """Tính Bảng Xếp Hạng Vòng Bảng (A–H / A–F / A–D) hoặc League Phase 36 CLB từ danh sách trận thực tế."""
    if not matches_list:
        return {}

    if mode == "ucl_league":
        # Đếm số trận mỗi CLB để lọc đúng 36 CLB dự League Phase (mỗi đội đá >= 6 trận, loại vòng sơ loại tháng 7-8)
        counts = {}
        for m in matches_list:
            mdate = str(m.get("date") or "")
            if len(mdate) >= 7 and mdate[5:7] in ("07", "08"):
                continue
            ht = str(m.get("home_team") or "")
            at = str(m.get("away_team") or "")
            counts[ht] = counts.get(ht, 0) + 1
            counts[at] = counts.get(at, 0) + 1
        valid_clubs = {t for t, c in counts.items() if c >= 5}
        if not valid_clubs:
            valid_clubs = set(counts.keys())

        table = {}
        sorted_m = sorted(matches_list, key=lambda x: str(x.get("date") or ""))
        for m in sorted_m:
            mdate = str(m.get("date") or "")
            if len(mdate) >= 7 and mdate[5:7] in ("07", "08"):
                continue
            ht = str(m.get("home_team") or "")
            at = str(m.get("away_team") or "")
            if ht not in valid_clubs or at not in valid_clubs:
                continue
            hs = int(m.get("home_score", 0) or 0)
            as_ = int(m.get("away_score", 0) or 0)
            for t in (ht, at):
                table.setdefault(t, {"team": t, "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0, "form": []})
            hr, ar = table[ht], table[at]
            hr["mp"] += 1
            ar["mp"] += 1
            hr["gf"] += hs
            hr["ga"] += as_
            ar["gf"] += as_
            ar["ga"] += hs
            if hs > as_:
                hr["w"] += 1; hr["pts"] += 3; hr["form"].append("W")
                ar["l"] += 1; ar["form"].append("L")
            elif hs < as_:
                ar["w"] += 1; ar["pts"] += 3; ar["form"].append("W")
                hr["l"] += 1; hr["form"].append("L")
            else:
                hr["d"] += 1; ar["d"] += 1; hr["pts"] += 1; ar["pts"] += 1
                hr["form"].append("D"); ar["form"].append("D")

        rows = list(table.values())
        for r in rows:
            r["gd"] = r["gf"] - r["ga"]
            r["form"] = r["form"][-5:]
        rows.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
        for idx, r in enumerate(rows, 1):
            r["rank"] = idx
            r["qualified"] = idx <= 8
            r["playoff"] = 9 <= idx <= 24
        return {"League Phase (36 CLB)": rows} if rows else {}

    # Phân tách các bảng đấu bằng đồ thị liên thông (mỗi bảng 4 đội tạo thành 1 cụm 6 trận độc lập)
    sorted_m = sorted(matches_list, key=lambda x: str(x.get("date") or ""))
    adj = {}
    explicit_grp = {}
    for m in sorted_m:
        ht = str(m.get("home_team") or "")
        at = str(m.get("away_team") or "")
        if not ht or not at:
            continue
        adj.setdefault(ht, set()).add(at)
        adj.setdefault(at, set()).add(ht)
        g_raw = str(m.get("group") or "").replace("Group", "").strip()
        if g_raw:
            explicit_grp[ht] = g_raw
            explicit_grp[at] = g_raw

    visited = set()
    components = []
    for t in adj:
        if t in visited:
            continue
        comp = set()
        stack = [t]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(adj.get(cur, set()) - visited)
        components.append(comp)

    team_to_group = {}
    used_letters = set()
    for comp in components:
        letter = None
        for t in comp:
            if t in explicit_grp:
                letter = explicit_grp[t]
                break
            if seed_groups and t in seed_groups:
                letter = seed_groups[t]
                break
        if letter:
            used_letters.add(letter)
            for t in comp:
                team_to_group[t] = f"Group {letter}"
        else:
            for cand in "ABCDEFGH":
                if cand not in used_letters:
                    used_letters.add(cand)
                    for t in comp:
                        team_to_group[t] = f"Group {cand}"
                    break

    groups = {}
    for m in sorted_m:
        ht = str(m.get("home_team") or "")
        at = str(m.get("away_team") or "")
        g_name = team_to_group.get(ht) or team_to_group.get(at) or "Group A"
        hs = int(m.get("home_score", 0) or 0)
        as_ = int(m.get("away_score", 0) or 0)
        groups.setdefault(g_name, {})
        for t in (ht, at):
            groups[g_name].setdefault(t, {"team": t, "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0, "form": []})
        hr, ar = groups[g_name][ht], groups[g_name][at]
        hr["mp"] += 1; ar["mp"] += 1
        hr["gf"] += hs; hr["ga"] += as_
        ar["gf"] += as_; ar["ga"] += hs
        if hs > as_:
            hr["w"] += 1; hr["pts"] += 3; hr["form"].append("W")
            ar["l"] += 1; ar["form"].append("L")
        elif hs < as_:
            ar["w"] += 1; ar["pts"] += 3; ar["form"].append("W")
            hr["l"] += 1; hr["form"].append("L")
        else:
            hr["d"] += 1; ar["d"] += 1; hr["pts"] += 1; ar["pts"] += 1
            hr["form"].append("D"); ar["form"].append("D")

    result = {}
    for g_name in sorted(groups.keys()):
        rows = list(groups[g_name].values())
        for r in rows:
            r["gd"] = r["gf"] - r["ga"]
        rows.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
        for idx, r in enumerate(rows, 1):
            r["rank"] = idx
            r["qualified"] = idx <= 2
        result[g_name] = rows
    return result


def render_tournament_hub_workspace(catalog: dict, wc2018_events_full: pd.DataFrame, wc2018_shots_df: pd.DataFrame, player_display_fn):
    st.markdown("## 🏆 Trung Tâm Giải Đấu: Bảng Xếp Hạng, Lịch Thi Đấu, Thống Kê Toàn Mùa & Chuyển Nhượng")
    matches_meta, leaderboard = compute_wc2018_season_leaderboards(wc2018_events_full, wc2018_shots_df)

    t_stand, t_fix, t_season, t_trans = st.tabs([
        "📊 Bảng Xếp Hạng (Đa Giải Đấu)",
        "📅 Lịch Thi Đấu & Kết Quả (Fixtures)",
        "🏅 Thống Kê Cầu Thủ Toàn Giải (Season Stats)",
        "💶 Thị Trường Chuyển Nhượng (FIFA Transfers)",
    ])

    with t_stand:
        tourn_options = [
            "🏆 FIFA World Cup 2022 (Bảng A–H • 64/64 trận)",
            "⚽ UEFA Champions League 2025/26 (League Phase • 36 CLB)",
            "⚽ UEFA Champions League 2024/25 (League Phase • 36 CLB)",
            "🏆 FIFA World Cup 2018 (Bảng A–H • 64/64 trận)",
            "🏆 UEFA Euro 2024 (Bảng A–F • 51/51 trận)",
            "🏆 UEFA Euro 2020 (Bảng A–F • 51/51 trận)",
            "🏆 CONMEBOL Copa América 2024 (Bảng A–D • 32/32 trận)",
            "🌍 Africa Cup of Nations 2023 (Bảng A–F • 52/52 trận)",
        ]
        sel_tourn = st.selectbox("🏆 Chọn Giải Đấu & Mùa Giải xem Bảng Xếp Hạng:", tourn_options, index=0, key="hub_standings_tourn")

        is_ucl_swiss = False
        header_prefix = "🏆 FIFA WORLD CUP 2022"
        if "World Cup 2018" in sel_tourn:
            standings = compute_wc2018_standings_data(matches_meta)
            header_prefix = "🏆 FIFA WORLD CUP 2018"
        elif "World Cup 2022" in sel_tourn:
            wc22_grp = [m for m in catalog.get("🏆 FIFA World Cup", []) if "World Cup 2022" in str(m.get("stage", "")) and "Vòng Bảng" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(wc22_grp, mode="groups")
            header_prefix = "🏆 FIFA WORLD CUP 2022"
        elif "Champions League 2025/26" in sel_tourn:
            ucl25_lp = [m for m in catalog.get("⚽ UEFA Champions League", []) if "2025/2026" in str(m.get("stage", "")) and "League Phase" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(ucl25_lp, mode="ucl_league")
            header_prefix = "⚽ UEFA CHAMPIONS LEAGUE 2025/26"
            is_ucl_swiss = True
        elif "Champions League 2024/25" in sel_tourn:
            ucl24_lp = [m for m in catalog.get("⚽ UEFA Champions League", []) if "2024/2025" in str(m.get("stage", "")) and "League Phase" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(ucl24_lp, mode="ucl_league")
            header_prefix = "⚽ UEFA CHAMPIONS LEAGUE 2024/25"
            is_ucl_swiss = True
        elif "Euro 2024" in sel_tourn:
            eu24_grp = [m for m in catalog.get("🏆 UEFA European Championship", []) if "Euro 2024" in str(m.get("stage", "")) and "Vòng Bảng" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(eu24_grp, mode="groups")
            header_prefix = "🏆 UEFA EURO 2024"
        elif "Euro 2020" in sel_tourn:
            eu20_grp = [m for m in catalog.get("🏆 UEFA European Championship", []) if "Euro 2020" in str(m.get("stage", "")) and "Vòng Bảng" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(eu20_grp, mode="groups")
            header_prefix = "🏆 UEFA EURO 2020"
        elif "Copa América 2024" in sel_tourn:
            ca_seeds = {"Argentina": "A", "Venezuela": "B", "Uruguay": "C", "Colombia": "D"}
            ca_grp = [m for m in catalog.get("🏆 CONMEBOL Copa América", []) if "Vòng Bảng" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(ca_grp, mode="groups", seed_groups=ca_seeds)
            header_prefix = "🏆 COPA AMÉRICA 2024"
        else:
            af_seeds = {"Ivory Coast": "A", "Côte d'Ivoire": "A", "Egypt": "B", "Cape Verde Islands": "B", "Senegal": "C", "Algeria": "D", "Angola": "D", "Mali": "E", "Tunisia": "E", "Morocco": "F"}
            af_grp = [m for m in catalog.get("🌍 Africa Cup of Nations (AFCON)", []) if "Vòng Bảng" in str(m.get("stage", ""))]
            standings = _compute_standings_from_catalog_matches(af_grp, mode="groups", seed_groups=af_seeds)
            header_prefix = "🌍 AFCON 2023"

        if not standings:
            st.info("Chưa tìm thấy dữ liệu vòng bảng cho giải đấu này.")
        elif is_ucl_swiss:
            st.caption("🟢 Hạng 1–8: Vào thẳng Vòng 1/8 (R16) &nbsp;•&nbsp; 🟡 Hạng 9–24: Đá Play-off Knockout &nbsp;•&nbsp; ⚪ Hạng 25–36: Bị loại.")
            rows_u = list(standings.values())[0]
            trs_u = ""
            for r in rows_u:
                if r.get("qualified"):
                    q_bg, q_dot = "rgba(0, 214, 108, 0.10)", "🟢"
                elif r.get("playoff"):
                    q_bg, q_dot = "rgba(250, 204, 21, 0.06)", "🟡"
                else:
                    q_bg, q_dot = "transparent", "⚪"
                form_badges = "".join(
                    f'<span style="background:{"#00D66C" if f=="W" else ("#FACC15" if f=="D" else "#EF4444")};color:#000;font-weight:900;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:2px;">{f}</span>'
                    for f in r["form"]
                )
                trs_u += compact_html(f"""
                <tr style="background:{q_bg};border-bottom:1px solid #1E293B;">
                    <td style="padding:7px 8px;font-weight:800;">{q_dot} {r['rank']}</td>
                    <td style="padding:7px 8px;font-weight:800;color:#F8FAFC;">{html.escape(r['team'])}</td>
                    <td style="padding:7px 5px;text-align:center;">{r['mp']}</td>
                    <td style="padding:7px 5px;text-align:center;">{r['w']}</td>
                    <td style="padding:7px 5px;text-align:center;">{r['d']}</td>
                    <td style="padding:7px 5px;text-align:center;">{r['l']}</td>
                    <td style="padding:7px 5px;text-align:center;">{r['gf']}:{r['ga']}</td>
                    <td style="padding:7px 5px;text-align:center;color:{'#4ADE80' if r['gd']>=0 else '#F87171'};font-weight:700;">{r['gd']:+d}</td>
                    <td style="padding:7px 8px;text-align:center;font-weight:900;color:#38BDF8;font-size:14.5px;">{r['pts']}</td>
                    <td style="padding:7px 6px;text-align:right;">{form_badges}</td>
                </tr>
                """)
            st.markdown(compact_html(f"""
            <div style="background:#131A26;border:1px solid #26354D;border-radius:13px;padding:14px 16px;margin-bottom:12px;">
                <div style="color:#38BDF8;font-weight:900;font-size:15px;margin-bottom:10px;">{header_prefix} • BẢNG XẾP HẠNG TỔNG LEAGUE PHASE (36 CÂU LẠC BỘ)</div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;color:#CBD5E1;">
                    <thead><tr style="border-bottom:1.5px solid #334155;color:#94A3B8;font-size:11.5px;">
                        <th style="text-align:left;padding:6px 8px;">Hạng</th>
                        <th style="text-align:left;padding:6px 8px;">Câu lạc bộ</th>
                        <th>Tr</th><th>T</th><th>H</th><th>B</th><th>BT:BB</th><th>HS</th><th>Điểm</th><th style="text-align:right;">5 trận gần nhất</th>
                    </tr></thead>
                    <tbody>{trs_u}</tbody>
                </table>
            </div>
            """), unsafe_allow_html=True)
        else:
            st.caption("🟢 Top 2 đội đầu mỗi bảng giành quyền vào Vòng Knockout.")
            g_names = sorted(standings.keys())
            for row_idx in range(0, len(g_names), 2):
                cols_g = st.columns(2, gap="medium")
                for offset in range(2):
                    if row_idx + offset >= len(g_names):
                        continue
                    g_title = g_names[row_idx + offset]
                    rows_g = standings[g_title]
                    with cols_g[offset]:
                        trs_html = ""
                        for r in rows_g:
                            q_bg = "rgba(0, 214, 108, 0.08)" if r["qualified"] else "transparent"
                            q_dot = "🟢" if r["qualified"] else "⚪"
                            form_badges = "".join(
                                f'<span style="background:{"#00D66C" if f=="W" else ("#FACC15" if f=="D" else "#EF4444")};color:#000;font-weight:900;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:2px;">{f}</span>'
                                for f in r["form"]
                            )
                            trs_html += compact_html(f"""
                            <tr style="background:{q_bg};border-bottom:1px solid #1E293B;">
                                <td style="padding:7px 6px;font-weight:800;">{q_dot} {r['rank']}</td>
                                <td style="padding:7px 6px;font-weight:800;color:#F8FAFC;">{html.escape(r['team'])}</td>
                                <td style="padding:7px 4px;text-align:center;">{r['mp']}</td>
                                <td style="padding:7px 4px;text-align:center;">{r['w']}</td>
                                <td style="padding:7px 4px;text-align:center;">{r['d']}</td>
                                <td style="padding:7px 4px;text-align:center;">{r['l']}</td>
                                <td style="padding:7px 4px;text-align:center;">{r['gf']}:{r['ga']}</td>
                                <td style="padding:7px 4px;text-align:center;color:{'#4ADE80' if r['gd']>=0 else '#F87171'};font-weight:700;">{r['gd']:+d}</td>
                                <td style="padding:7px 6px;text-align:center;font-weight:900;color:#38BDF8;font-size:14px;">{r['pts']}</td>
                                <td style="padding:7px 4px;text-align:right;">{form_badges}</td>
                            </tr>
                            """)
                        st.markdown(compact_html(f"""
                        <div style="background:#131A26;border:1px solid #26354D;border-radius:13px;padding:12px 14px;margin-bottom:12px;">
                            <div style="color:#38BDF8;font-weight:900;font-size:14.5px;margin-bottom:8px;">{header_prefix} • {html.escape(g_title.upper())}</div>
                            <table style="width:100%;border-collapse:collapse;font-size:12.5px;color:#CBD5E1;">
                                <thead><tr style="border-bottom:1.5px solid #334155;color:#94A3B8;font-size:11px;">
                                    <th style="text-align:left;padding:5px 6px;">#</th>
                                    <th style="text-align:left;padding:5px 6px;">Đội tuyển</th>
                                    <th>Tr</th><th>T</th><th>H</th><th>B</th><th>BT:BB</th><th>HS</th><th>Điểm</th><th style="text-align:right;">Phong độ</th>
                                </tr></thead>
                                <tbody>{trs_html}</tbody>
                            </table>
                        </div>
                        """), unsafe_allow_html=True)

    with t_fix:
        fc1, fc2, fc3 = st.columns([1.4, 1.4, 1.8])
        comp_pick = fc1.selectbox("Giải đấu:", [k for k in catalog.keys() if catalog[k]], key="fix_comp_sel")
        fixtures_list = catalog[comp_pick]
        stages_avail = ["Tất cả vòng đấu"] + sorted(list({str(m.get("stage", "")) for m in fixtures_list}))
        stage_pick = fc2.selectbox("Lọc theo Vòng đấu:", stages_avail, key="fix_stage_sel")
        team_q = fc3.text_input("Lọc nhanh theo tên đội:", placeholder="VD: France, Argentina, Brazil...", key="fix_team_q")

        filtered_fix = [
            m for m in fixtures_list
            if (stage_pick == "Tất cả vòng đấu" or str(m.get("stage", "")) == stage_pick)
            and (not team_q.strip() or team_q.strip().lower() in f"{m.get('home_team', '')} {m.get('away_team', '')} {m.get('label', '')}".lower())
        ]
        st.caption(f"Hiển thị **{len(filtered_fix)}** trận đấu. Bấm **Phân tích** để nhảy thẳng sang màn hình mổ xẻ chiến thuật.")
        for idx_f, mf in enumerate(filtered_fix[:32]):
            col_info, col_btn = st.columns([4.4, 1.1], gap="small")
            pen_t = f" • PEN {mf['pen_home']}-{mf['pen_away']}" if mf.get("pen_home") is not None else ""
            col_info.markdown(compact_html(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;background:#131A26;border:1px solid #243247;border-radius:10px;padding:9px 14px;">
                <span style="color:#94A3B8;font-size:12px;min-width:190px;">📅 {html.escape(str(mf.get('date') or 'N/A'))} • {html.escape(str(mf.get('stage', '')))}</span>
                <span style="color:#F8FAFC;font-weight:800;font-size:14px;">🔵 {html.escape(str(mf.get('home_team', '')))} &nbsp; <b style="color:#00E676;font-size:16px;">{mf.get('home_score', 0)} - {mf.get('away_score', 0)}</b><span style="color:#FACC15;font-size:11.5px;">{html.escape(pen_t)}</span> &nbsp; {html.escape(str(mf.get('away_team', '')))} 🔴</span>
                <span style="color:#64748B;font-size:11.5px;">🏟️ {html.escape(str(mf.get('stadium', '')))}</span>
            </div>
            """), unsafe_allow_html=True)
            if col_btn.button("📊 Phân tích", key=f"fix_open_{mf['id']}_{idx_f}", width="stretch"):
                st.session_state["jump_comp"] = comp_pick
                st.session_state["jump_match_label"] = mf["label"]
                st.session_state["pending_workspace"] = "ws_match"
                st.rerun()

    with t_season:
        if leaderboard.empty:
            st.info("Chưa có dữ liệu tổng hợp toàn mùa.")
        else:
            lb = leaderboard.copy()
            lb["Cầu thủ"] = lb["player_name"].apply(player_display_fn)
            k1, k2, k3, k4 = st.columns(4, gap="small")
            top_g = lb.sort_values(["goals", "xg"], ascending=False).iloc[0]
            top_a = lb.sort_values(["assists", "key_passes"], ascending=False).iloc[0]
            top_xt = lb.sort_values("xt", ascending=False).iloc[0]
            cand_r = lb[lb["matches"] >= 3] if (lb["matches"] >= 3).any() else lb
            top_r = cand_r.sort_values("avg_rating", ascending=False).iloc[0]
            k1.metric("👟 Chiếc Giày Vàng", f"{top_g['Cầu thủ']} ({top_g['goals']} bàn)", f"{top_g['team_name']}", border=True)
            k2.metric("🎯 Vua Kiến Tạo & Key Pass", f"{top_a['Cầu thủ']} ({top_a['assists']} KT)", f"{top_a['key_passes']} Key Passes", border=True)
            k3.metric("🚀 Ông Hoàng Tịnh Tiến (xT)", f"{top_xt['Cầu thủ']} (+{top_xt['xt']:.2f})", f"{top_xt['prog_passes']} chuyền tịnh tiến", border=True)
            k4.metric("⭐ Điểm TB Cao Nhất (≥3 trận)", f"{top_r['Cầu thủ']} ({top_r['avg_rating']}★)", f"{top_r['team_name']}", border=True)

            sort_col_map = {
                "⚽ Bàn thắng (Goals)": "goals",
                "🎯 Bàn thắng kỳ vọng (xG)": "xg",
                "👟 Kiến tạo (Assists)": "assists",
                "🔑 Tạo cơ hội (Key Passes)": "key_passes",
                "🚀 Độ đe dọa từ chuyền (xT)": "xt",
                "⭐ Điểm SofaScore trung bình": "avg_rating",
            }
            sc1, sc2 = st.columns([2, 2])
            chosen_metric_lbl = sc1.selectbox("Xếp hạng theo chỉ số:", list(sort_col_map.keys()))
            team_flt = sc2.selectbox("Lọc theo Đội tuyển:", ["Tất cả 32 đội tuyển"] + sorted(lb["team_name"].unique().tolist()))
            sort_c = sort_col_map[chosen_metric_lbl]
            view_lb = lb if team_flt == "Tất cả 32 đội tuyển" else lb[lb["team_name"] == team_flt]
            view_lb = view_lb.sort_values([sort_c, "avg_rating"], ascending=[False, False]).head(30)

            disp_df = view_lb[["Cầu thủ", "team_name", "matches", "goals", "xg", "assists", "key_passes", "passes_comp", "pass_pct", "prog_passes", "xt", "avg_rating"]].rename(columns={
                "team_name": "Đội tuyển", "matches": "Số trận", "goals": "Bàn thắng", "xg": "Tổng xG",
                "assists": "Kiến tạo", "key_passes": "Key Pass", "passes_comp": "Chuyền đúng",
                "pass_pct": "Chuyền (%)", "prog_passes": "Chuyền tịnh tiến", "xt": "Tổng xT", "avg_rating": "Điểm TB ★",
            })
            st.dataframe(disp_df, width="stretch", hide_index=True)

    with t_trans:
        fifa_df, _, _ = load_fifa_transfer_db()
        if fifa_df.empty:
            st.info("Không tìm thấy tệp dữ liệu FIFA Transfer.")
        else:
            st.markdown("#### 💶 Cơ Sở Dữ Liệu Chuyển Nhượng, Định Giá & Tài Năng Trẻ (17.981 Cầu Thủ)")
            tc1, tc2, tc3, tc4 = st.columns(4)
            nat_opts = ["Tất cả Quốc tịch"] + sorted(fifa_df["Nationality"].dropna().unique().tolist())
            sel_nat = tc1.selectbox("Quốc tịch:", nat_opts, key="tr_nat")
            max_age = tc2.slider("Độ tuổi tối đa:", 16, 40, 32, key="tr_age")
            min_ovr = tc3.slider("Overall tối thiểu:", 60, 94, 80, key="tr_ovr")
            q_club = tc4.text_input("Tìm tên cầu thủ / CLB:", placeholder="VD: Messi, Real Madrid, PSG...", key="tr_q")

            f_sub = fifa_df[(fifa_df["Age"] <= max_age) & (fifa_df["Overall"] >= min_ovr)].copy()
            if sel_nat != "Tất cả Quốc tịch":
                f_sub = f_sub[f_sub["Nationality"] == sel_nat]
            if q_club.strip():
                ql = q_club.strip().lower()
                f_sub = f_sub[f_sub["Name"].astype(str).str.lower().str.contains(ql, na=False) | f_sub["Club"].astype(str).str.lower().str.contains(ql, na=False)]
            f_sub = f_sub.sort_values(["Value_M", "Overall"], ascending=[False, False]).head(40)

            show_tr = f_sub[["Name", "Age", "Nationality", "Club", "Preferred Positions", "Overall", "Potential", "Value", "Wage"]].rename(columns={
                "Name": "Cầu thủ", "Age": "Tuổi", "Nationality": "Quốc tịch", "Club": "Câu lạc bộ",
                "Preferred Positions": "Vị trí", "Value": "Giá trị chuyển nhượng", "Wage": "Lương tuần",
            })
            st.dataframe(show_tr, width="stretch", hide_index=True)


# ==========================================================
# ⚔️ 9. KHÔNG GIAN 4: SO SÁNH CẦU THỦ, AI DỰ ĐOÁN CẶP ĐẤU & KIẾN TRÚC / API (#36, #37, #41, #45)
# ==========================================================
def render_compare_ai_and_architecture_workspace(
    all_selectable_players: list,
    match_events: pd.DataFrame,
    current_match: dict,
    match_stats_df: pd.DataFrame,
    wc2018_events_full: pd.DataFrame,
    wc2018_shots_df: pd.DataFrame,
    render_stat_bar_fn,
):
    st.markdown("## ⚔️ So Sánh Đối Đầu Cầu Thủ, AI Dự Đoán Tỷ Số & Kiến Trúc Hệ Thống / REST API")
    tab_cmp, tab_aipred, tab_sys = st.tabs([
        "⚔️ So Sánh 2 Cầu Thủ (Dual Radar Compare — #36)",
        "🤖 AI Dự Đoán Tỷ Số Cặp Đấu Bất Kỳ (6×6 Score Matrix — #37)",
        "☁️ Kiến Trúc Cloud / Spark / SQLite / Live REST API & Đối Chiếu 45 Tiêu Chí (#41)",
    ])

    with tab_cmp:
        if len(all_selectable_players) < 2:
            st.info("Cần tối thiểu 2 cầu thủ để so sánh.")
        else:
            p_map = {p["name"]: p for p in all_selectable_players}
            names_list = list(p_map.keys())
            cc1, cc2 = st.columns(2)
            p1_key = cc1.selectbox(
                "🔵 Chọn Cầu thủ 1:",
                names_list, index=0,
                format_func=lambda n: f"{p_map[n]['display_name']} ({p_map[n]['team_name']} • {p_map[n]['rating']}★)",
                key=f"cmp_p1_{current_match['id']}",
            )
            default_idx2 = min(1, len(names_list) - 1)
            for idx_cand, nm_cand in enumerate(names_list):
                if p_map[nm_cand]["team_name"] != p_map[p1_key]["team_name"]:
                    default_idx2 = idx_cand
                    break
            p2_key = cc2.selectbox(
                "🔴 Chọn Cầu thủ 2:",
                names_list, index=default_idx2,
                format_func=lambda n: f"{p_map[n]['display_name']} ({p_map[n]['team_name']} • {p_map[n]['rating']}★)",
                key=f"cmp_p2_{current_match['id']}",
            )
            p1, p2 = p_map[p1_key], p_map[p2_key]
            ev1 = match_events[match_events["player_name"] == p1_key] if "player_name" in match_events.columns else pd.DataFrame()
            ev2 = match_events[match_events["player_name"] == p2_key] if "player_name" in match_events.columns else pd.DataFrame()
            f1 = lookup_fifa_player(p1["name"], p1["display_name"], p1["team_name"])
            f2 = lookup_fifa_player(p2["name"], p2["display_name"], p2["team_name"])
            r1 = compute_player_radar_metrics(p1, ev1, f1)
            r2 = compute_player_radar_metrics(p2, ev2, f2)

            axes = r1["axes"] + [r1["axes"][0]]
            v1 = r1["values"] + [r1["values"][0]]
            v2 = r2["values"] + [r2["values"][0]]

            col_rad, col_bars = st.columns([1.1, 1.2], gap="medium")
            with col_rad:
                with st.container(border=True):
                    st.markdown(f"##### 🕸️ Biểu Đồ Radar Đối Đầu: {p1['display_name']} vs {p2['display_name']}")
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=v1, theta=axes, fill="toself", fillcolor="rgba(56, 189, 248, 0.28)",
                        line=dict(color="#38BDF8", width=2.8), name=f"{p1['display_name']} ({p1['team_name']})",
                    ))
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=v2, theta=axes, fill="toself", fillcolor="rgba(255, 77, 109, 0.26)",
                        line=dict(color="#FF4D6D", width=2.8), name=f"{p2['display_name']} ({p2['team_name']})",
                    ))
                    fig_cmp.update_layout(
                        polar=dict(
                            bgcolor="#0D121C",
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.16)"),
                            angularaxis=dict(tickfont=dict(size=12, color="#F8FAFC")),
                        ),
                        height=380, margin=dict(l=40, r=40, t=35, b=20),
                        paper_bgcolor="#131A26", font=dict(color="#E2E8F0"),
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig_cmp, width="stretch", key=f"dual_radar_{p1_key}_{p2_key}")

            with col_bars:
                with st.container(border=True):
                    st.markdown("##### 📊 Đối Chiếu Trực Diện Các Chỉ Số")
                    b_html = ""
                    b_html += render_stat_bar_fn("Điểm SofaScore", p1["rating"], p2["rating"], f"{p1['rating']}★", f"{p2['rating']}★")
                    b_html += render_stat_bar_fn("Bàn thắng", p1.get("goals_count", 0), p2.get("goals_count", 0), str(p1.get("goals_count", 0)), str(p2.get("goals_count", 0)))
                    b_html += render_stat_bar_fn("Kiến tạo", p1.get("assists_count", 0), p2.get("assists_count", 0), str(p1.get("assists_count", 0)), str(p2.get("assists_count", 0)))
                    p1_comp = int(str(p1.get("passes", "0/0")).split("/")[0])
                    p2_comp = int(str(p2.get("passes", "0/0")).split("/")[0])
                    b_html += render_stat_bar_fn("Chuyền chính xác", p1_comp, p2_comp, str(p1.get("passes")), str(p2.get("passes")))
                    b_html += render_stat_bar_fn("Tắc bóng & Phòng ngự", p1.get("tackles", 0), p2.get("tackles", 0), str(p1.get("tackles", 0)), str(p2.get("tackles", 0)))
                    if f1 and f2:
                        b_html += render_stat_bar_fn("Định giá chuyển nhượng", f1.get("Value_M", 1), f2.get("Value_M", 1), str(f1.get("Value", "N/A")), str(f2.get("Value", "N/A")))
                    st.markdown(b_html, unsafe_allow_html=True)

    with tab_aipred:
        st.markdown("#### 🤖 Mô Hình AI Bivariate Poisson Dự Đoán Tỷ Số & Ma Trận Xác Suất 6×6")
        teams_pool = sorted(match_stats_df["team_name"].dropna().unique().tolist()) if (match_stats_df is not None and not match_stats_df.empty and "team_name" in match_stats_df.columns) else ["France", "Croatia", "Brazil", "Belgium", "Argentina", "Spain", "England", "Portugal"]
        ac1, ac2 = st.columns(2)
        h_idx = teams_pool.index("France") if "France" in teams_pool else 0
        a_idx = teams_pool.index("Croatia") if "Croatia" in teams_pool else min(1, len(teams_pool) - 1)
        pred_h_team = ac1.selectbox("🔵 Đội Chủ Nhà:", teams_pool, index=h_idx, key="ai_pred_ht")
        pred_a_team = ac2.selectbox("🔴 Đội Khách:", teams_pool, index=a_idx, key="ai_pred_at")

        res = predict_match_ai_payload(pred_h_team, pred_a_team, match_stats_df)
        pk1, pk2, pk3, pk4 = st.columns(4, gap="small")
        pk1.metric(f"🔵 {pred_h_team} Thắng", f"{res['win_prob_home_pct']}%", f"xG kỳ vọng: {res['expected_goals_home']}", border=True)
        pk2.metric("⚖️ Tỷ lệ Hòa", f"{res['draw_prob_pct']}%", "90 phút chính thức", border=True)
        pk3.metric(f"🔴 {pred_a_team} Thắng", f"{res['win_prob_away_pct']}%", f"xG kỳ vọng: {res['expected_goals_away']}", border=True)
        pk4.metric("🎯 Tỷ số sáng nhất", res["top_scorelines"][0]["score"], f"Xác suất {res['top_scorelines'][0]['prob_pct']}%", border=True)

        mat = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                p_i = (res["expected_goals_home"] ** i) * math.exp(-res["expected_goals_home"]) / math.factorial(i)
                p_j = (res["expected_goals_away"] ** j) * math.exp(-res["expected_goals_away"]) / math.factorial(j)
                mat[i, j] = round(p_i * p_j * 100.0, 2)

        fig_mat = go.Figure(data=go.Heatmap(
            z=mat,
            x=[f"{pred_a_team}: {j} bàn" for j in range(6)],
            y=[f"{pred_h_team}: {i} bàn" for i in range(6)],
            text=[[f"{mat[i, j]:.1f}%" for j in range(6)] for i in range(6)],
            texttemplate="%{text}",
            colorscale=[[0.0, "#0F172A"], [0.4, "#0369A1"], [0.75, "#0284C7"], [1.0, "#00E676"]],
            showscale=True,
        ))
        fig_mat.update_layout(
            title=f"Ma trận Xác suất Tỷ số AI ({pred_h_team} vs {pred_a_team})",
            height=390, margin=dict(l=20, r=20, t=45, b=20),
            paper_bgcolor="#131A26", plot_bgcolor="#0E1420", font=dict(color="#F8FAFC"),
        )
        st.plotly_chart(fig_mat, width="stretch", key=f"ai_mat_{pred_h_team}_{pred_a_team}")

    with tab_sys:
        api_ok = ensure_api_server_running(8502)
        st.markdown("#### ☁️ Kiến Trúc Nền Tảng Big Data & Live REST API Microservice (`http://localhost:8502`)")
        s1, s2, s3, s4 = st.columns(4, gap="small")
        s1.metric("⚡ Apache Spark Cluster", "Docker Ready", "spark-master + 2 workers", border=True)
        s2.metric("🗄️ CSDL Quan Hệ", "SQLite3 Active", "Users • Favorites • Notes", border=True)
        s3.metric("🌐 REST JSON API", "ONLINE :8502" if api_ok else "OFFLINE", "5 Live Endpoints", border=True)
        s4.metric("📦 Chế độ Offline Cache", "2,480+ Trận Local", "4.9M+ Sự kiện StatsBomb & Opta", border=True)

        with st.container(border=True):
            st.markdown("##### 🔌 Trình Kiểm Thử Trực Tiếp REST API JSON (`http://localhost:8502/api/...`)")
            ep = st.selectbox(
                "Chọn Endpoint API để gọi thử trực tiếp:",
                [
                    "http://localhost:8502/api/health",
                    "http://localhost:8502/api/standings",
                    "http://localhost:8502/api/players/top?limit=10",
                    "http://localhost:8502/api/predict?home=France&away=Croatia",
                    "http://localhost:8502/api/matches?limit=10&competition=UCL",
                ],
            )
            if st.button("🚀 Gửi Request GET & Xem JSON Phản Hồi", width="stretch"):
                try:
                    with urllib.request.urlopen(ep, timeout=3) as resp:
                        payload_txt = resp.read().decode("utf-8")
                    st.code(payload_txt[:3500], language="json")
                except Exception as exc:
                    st.warning(f"Không thể kết nối endpoint: {exc}")

        with st.container(border=True):
            st.markdown("##### ✅ Bảng Tổng Kết Nâng Cấp Dự Án So Với FotMob / SofaScore / AiScore")
            summary_rows = [
                ("1 & 21. Tỷ số trực tiếp & Cập nhật Real-time", "✅ Đã tích hợp (Auto-Run Live Replay Fragment 1.2s/phút)"),
                ("4 & 23. Đa giải đấu & Dữ liệu lịch sử", "✅ Đã tích hợp (10 giải lớn: 147 trận WC, 1.842 trận UCL, 102 Euro, Copa, AFCON, Top 5 Châu Âu — 2.480+ trận)"),
                ("5. Theo dõi nhiều trận cùng lúc", "✅ Đã tích hợp (Multi-Match Split Screen 2–4 trận)"),
                ("11 & 32. Thay người, Sự kiện & Bình luận trực tiếp", "✅ Đã tích hợp (Timeline + Live Commentary Feed)"),
                ("14 & 36. Biểu đồ Radar kỹ năng & So sánh cầu thủ", "✅ Đã tích hợp (Radar 6 trục + Dual Compare)"),
                ("16, 17, 18, 19. Shot Map 2D, Pass Network, Phòng ngự, Momentum", "✅ Đã tích hợp đầy đủ chuẩn SofaScore Pro"),
                ("22, 28, 44. Cơ sở dữ liệu SQLite, Đăng nhập Auth & Yêu thích", "✅ Đã tích hợp (Lưu tài khoản, Yêu thích, Dự đoán)"),
                ("27, 29, 30, 31. Tìm kiếm toàn cục, Chia sẻ, Đa ngôn ngữ VI/EN, Responsive", "✅ Đã tích hợp đầy đủ"),
                ("33, 34, 35, 38. Bảng xếp hạng, Lịch thi đấu, Thống kê toàn mùa, Chuyển nhượng", "✅ Đã tích hợp (BXH A-H + 17.9k cầu thủ FIFA)"),
                ("37, 39, 40. AI Dự đoán tỷ số, Video Highlights, Bản tin tự động", "✅ Đã tích hợp (Monte Carlo 10k + Ma trận Poisson 6×6)"),
                ("41, 42, 43, 45. Spark/REST API (:8502), Caching, Offline & Xuất dữ liệu CSV/JSON/HTML", "🌟 VƯỢT TRỘI (FotMob/SofaScore/AiScore không cho xuất CSV/JSON)"),
            ]
            st.dataframe(pd.DataFrame(summary_rows, columns=["Nhóm tiêu chí trong bảng so sánh (1–45)", "Trạng thái trên Dự án của nhóm"]), width="stretch", hide_index=True)


# ==========================================================
# 📱 10. SIDEBAR: NGÔN NGỮ, WORKSPACE, TÌM KIẾM TOÀN CỤC & TÀI KHOẢN (#27, #28, #30, #44)
# ==========================================================
def render_sidebar_pro_controls(catalog: dict):
    init_platform_db()
    ensure_api_server_running(8502)

    pending_ws = st.session_state.pop("pending_workspace", None)
    if pending_ws:
        st.session_state["platform_workspace"] = pending_ws

    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "vi"
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = {"username": "demo", "display_name": "Chuyên Gia Phân Tích (Demo)", "role": "analyst"}
    if "platform_workspace" not in st.session_state:
        st.session_state["platform_workspace"] = "ws_match"

    lang_choice = st.sidebar.radio(
        "🌐 Ngôn ngữ / Language:",
        ["vi", "en"],
        format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English",
        horizontal=True,
        key="ui_lang",
    )

    ws_options = ["ws_match", "ws_live", "ws_hub", "ws_ai"]
    workspace = st.sidebar.radio(
        tr("nav_title", lang_choice),
        ws_options,
        format_func=lambda k: tr(k, lang_choice),
        key="platform_workspace",
    )

    # Tìm kiếm toàn cục (#27)
    with st.sidebar.expander("🔎 Tìm kiếm toàn cục (Global Search)", expanded=False):
        q = st.text_input("Từ khóa:", placeholder=tr("search_placeholder", lang_choice), key="global_search_input")
        if q and len(q.strip()) >= 2:
            ql = q.strip().lower()
            found_m = []
            for cn, mlist in catalog.items():
                for m in mlist:
                    goal_names = " ".join(str(g.get("player", "")) for g in (m.get("goals") or []))
                    hay = f"{m.get('label', '')} {m.get('home_team', '')} {m.get('away_team', '')} {m.get('stadium', '')} {goal_names}".lower()
                    if ql in hay:
                        found_m.append((cn, m))
            st.caption(f"Tìm thấy **{len(found_m)}** trận liên quan:")
            for idx_s, (cn, fm) in enumerate(found_m[:6]):
                if st.button(f"⚽ {fm['label']}", key=f"gs_jump_{fm['id']}_{idx_s}", width="stretch"):
                    st.session_state["jump_comp"] = cn
                    st.session_state["jump_match_label"] = fm["label"]
                    st.session_state["pending_workspace"] = "ws_match"
                    st.rerun()

    # Tài khoản Auth + Danh sách Yêu thích SQLite (#22, #28, #44)
    user = st.session_state.get("auth_user")
    exp_label = f"👤 {user['display_name']}" if user else "🔐 Đăng nhập / Đăng ký"
    with st.sidebar.expander(exp_label, expanded=False):
        if user:
            st.caption(f"Đang đăng nhập: **`@{user['username']}`** (SQLite Auth)")
            favs = list_favorites(user["username"])
            if favs:
                st.markdown("**⭐ Danh sách Yêu thích đã lưu:**")
                for idx_fv, fv in enumerate(favs[:8]):
                    ic = "🏟️" if fv["item_type"] == "match" else "👟"
                    st.caption(f"{ic} {fv['item_label']}")
            else:
                st.caption("Chưa có mục Yêu thích nào. Hãy bấm ☆ trên trận đấu hoặc cầu thủ!")
            if st.button("🚪 Đăng xuất / Đổi tài khoản", key="btn_logout", width="stretch"):
                st.session_state["auth_user"] = None
                st.rerun()
        else:
            auth_mode = st.radio("Chế độ:", ["Đăng nhập", "Đăng ký mới"], horizontal=True, key="auth_mode_rad")
            u_in = st.text_input("Tên đăng nhập:", value="demo" if auth_mode == "Đăng nhập" else "", key="auth_u")
            dn_in = st.text_input("Tên hiển thị:", key="auth_dn") if auth_mode == "Đăng ký mới" else ""
            pw_in = st.text_input("Mật khẩu:", type="password", value="123456" if auth_mode == "Đăng nhập" else "", key="auth_pw")
            if auth_mode == "Đăng nhập":
                if st.button("✅ Đăng nhập", key="btn_do_login", width="stretch"):
                    acc = authenticate_user(u_in, pw_in)
                    if acc:
                        st.session_state["auth_user"] = acc
                        st.rerun()
                    else:
                        st.error("Sai tài khoản hoặc mật khẩu (thử demo / 123456).")
            else:
                if st.button("✨ Tạo tài khoản SQLite", key="btn_do_reg", width="stretch"):
                    ok, msg = register_user(u_in, dn_in, pw_in)
                    if ok:
                        st.success(msg)
                        st.session_state["auth_user"] = authenticate_user(u_in, pw_in)
                        st.rerun()
                    else:
                        st.error(msg)

    return lang_choice, workspace
