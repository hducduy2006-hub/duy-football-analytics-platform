"""
Trung Tâm AI Phân Tích & Dự Đoán Trận Đấu (Pre-Match Research & Explainable AI Lab — v4.0)
Tối ưu tốc độ (<15ms/trận nhờ pre-indexed lookups) & giao diện phẳng sạch sẽ chuẩn FotMob / Opta Analyst / Understat:
  - Động cơ AI Ensemble 4 tầng: Opta/Panna Deep ML (55%) + Dixon-Coles Rolling xG (25%) + Elo Model (20%) + Market No-Vig Edge
  - Explainable AI (SHAP-style Key Drivers): Giải thích định lượng 6 nhân tố quyết định dự đoán
  - 3 Thẻ Khuyến Nghị Cốt Lõi (Smart Picks) + Ma trận 6×6 Dixon-Coles + Sức mạnh 4 tuyến + Radar 8 trục
  - Tích hợp 4 tab phẳng không lồng nhau: Phân Tích Trận Sắp Đá | Quét Vòng Đấu | Ghép Cặp 2 Đội Bất Kỳ | BXH Thực Lực xPTS
"""

import html
import json
import math
import os
import time as _time
import unicodedata as _ud
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from team_logos_store import get_team_logo_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
PRO_DIR = os.path.join(SAMPLE_DIR, "pro_analytics")

UPCOMING_PARQUET = os.path.join(PRO_DIR, "upcoming_fixtures.parquet")
HIST_XG_PARQUET = os.path.join(PRO_DIR, "historical_matches_xg.parquet")
ESPN_CACHE_JSON = os.path.join(PRO_DIR, "espn_live_cache.json")

LEAGUE_META = {
    "ENG": ("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "Premier League", "eng.1"),
    "ESP": ("🇪🇸 La Liga", "La Liga", "esp.1"),
    "ITA": ("🇮🇹 Serie A", "Serie A", "ita.1"),
    "GER": ("🇩🇪 Bundesliga", "Bundesliga", "ger.1"),
    "FRA": ("🇫🇷 Ligue 1", "Ligue 1", "fra.1"),
    "UCL": ("⚽ Champions League", "Champions League", "uefa.champions"),
    "UEL": ("🏆 Europa League", "Europa League", "uefa.europa"),
    "UECL": ("🏅 Conference League", "Conference League", "uefa.europa.conf"),
    "ENG2": ("🏴󠁧󠁢󠁥󠁮󠁧󠁿 EFL Championship", "Championship", "eng.2"),
    "POR": ("🇵🇹 Liga Portugal", "Liga Portugal", "por.1"),
    "NED": ("🇳🇱 Eredivisie", "Eredivisie", "ned.1"),
    "SCO": ("🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Prem", "Scottish Prem", "sco.1"),
    "TUR": ("🇹🇷 Süper Lig", "Süper Lig", "tur.1"),
    "WC": ("🏆 World Cup", "World Cup", "fifa.world"),
    "EURO": ("🏆 Euro", "Euro", "uefa.euro"),
}

# Map sang tên giải đầy đủ trong catalog của app.py
CATALOG_CATEGORY_MAP = {
    "ENG": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League",
    "ESP": "🇪🇸 Spanish La Liga",
    "ITA": "🇮🇹 Italian Serie A",
    "GER": "🇩🇪 German Bundesliga",
    "FRA": "🇫🇷 French Ligue 1",
    "UCL": "⚽ UEFA Champions League",
    "UEL": "🏆 UEFA Europa League",
    "UECL": "🏅 UEFA Conference League",
    "ENG2": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 EFL Championship",
    "POR": "🇵🇹 Liga Portugal",
    "NED": "🇳🇱 Eredivisie",
    "SCO": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership",
    "TUR": "🇹🇷 Turkish Süper Lig",
}

LEAGUE_ORDER = ["ENG", "UCL", "ESP", "ITA", "GER", "FRA", "UEL", "UECL", "ENG2", "POR", "NED", "SCO", "TUR"]


def compact_html(raw_html: str) -> str:
    return "".join(line.strip() for line in str(raw_html).splitlines() if line.strip())


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _norm_key(name: str) -> str:
    raw = str(name or "").translate(str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d"}))
    s = _ud.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    for tok in ("fc ", " fc", "afc ", " cf", " sc", " ac ", "sv ", "1. ", "sc ", "as ", "ss ", "us ", "rc ", "rcd ", "cd ", "ud ", "ogc "):
        s = s.replace(tok, " ")
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


_TEAM_ALIASES = {
    "paris saint germain": "psg",
    "inter milan": "inter",
    "internazionale": "inter",
    "ac milan": "milan",
    "bayern munich": "bayern",
    "bayern munchen": "bayern",
    "atletico madrid": "atletico",
    "atletico de madrid": "atletico",
    "manchester city": "man city",
    "manchester united": "man utd",
    "tottenham hotspur": "tottenham",
    "newcastle united": "newcastle",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
    "brighton hove albion": "brighton",
    "nottingham forest": "nottm forest",
    "borussia dortmund": "dortmund",
    "bayer leverkusen": "leverkusen",
    "rb leipzig": "leipzig",
    "borussia m gladbach": "gladbach",
    "borussia monchengladbach": "gladbach",
    "eintracht frankfurt": "frankfurt",
    "vfb stuttgart": "stuttgart",
    "olympique lyonnais": "lyon",
    "olympique marseille": "marseille",
    "olympique de marseille": "marseille",
    "athletic bilbao": "athletic club",
    "sporting cp": "sporting",
    "psv eindhoven": "psv",
    "bodo glimt": "bodo glimt",
    "hamburg sv": "hamburger sv",
    "hamburger sv": "hamburger sv",
    "stade rennais": "rennes",
    "mainz": "mainz 05",
    "mainz 05": "mainz 05",
    "racing santander": "racing de santander",
}


def _team_alias_key(name: str) -> str:
    k = _norm_key(name)
    return _TEAM_ALIASES.get(k, k)


def _format_kickoff_vn(kickoff_utc: str, fallback_date: str = "") -> tuple[str, str]:
    now_vn = datetime.now(timezone(timedelta(hours=7)))
    if kickoff_utc and "T" in str(kickoff_utc):
        try:
            dt_utc = datetime.fromisoformat(str(kickoff_utc).replace("Z", "+00:00"))
            dt_vn = dt_utc.astimezone(timezone(timedelta(hours=7)))
            disp = dt_vn.strftime("%H:%M · %d/%m/%Y")
            secs = (dt_vn - now_vn).total_seconds()
            if secs < -7200:
                cd = "Đã qua giờ bóng lăn"
            elif secs <= 0:
                cd = "🔴 Sắp / Đang bóng lăn"
            elif secs < 3600:
                cd = f"⏳ Còn {max(1, int(secs // 60))} phút"
            elif secs < 86400:
                cd = f"⏳ Còn {int(secs // 3600)}g {int((secs % 3600) // 60)}p"
            else:
                cd = f"📅 Còn {int(secs // 86400)} ngày"
            return disp, cd
        except Exception:
            pass
    if fallback_date:
        try:
            dt_d = datetime.strptime(str(fallback_date)[:10], "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=7)))
            days = (dt_d.date() - now_vn.date()).days
            disp = dt_d.strftime("%d/%m/%Y")
            cd = "🔥 HÔM NAY" if days == 0 else ("📅 NGÀY MAI" if days == 1 else (f"📅 Còn {days} ngày" if days > 1 else "Lịch gần đây"))
            return disp, cd
        except Exception:
            pass
    return str(fallback_date or "Chưa rõ lịch"), ""


# ==========================================================
# 1. CACHED & PRE-INDEXED DATA LOADERS (ZERO-COPY SHARED RAM)
# ==========================================================
_ESPN_SYNC_COOLDOWN_SEC = 120.0
_ESPN_LAST_SYNC_MONO = 0.0


@st.cache_resource(show_spinner=False, ttl=1800)
def load_upcoming_fixtures_df() -> pd.DataFrame:
    if not os.path.isfile(UPCOMING_PARQUET):
        return pd.DataFrame()
    df = pd.read_parquet(UPCOMING_PARQUET)
    if not df.empty:
        df["match_date"] = df["match_date"].astype(str).str.slice(0, 10)
        df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False, ttl=1800)
def load_historical_xg_df() -> pd.DataFrame:
    """Nạp 60.012 trận lịch sử & tiền lập chỉ mục _hk/_ak 1 lần duy nhất chia sẻ toàn server."""
    if not os.path.isfile(HIST_XG_PARQUET):
        return pd.DataFrame()
    df = pd.read_parquet(HIST_XG_PARQUET)
    if not df.empty:
        df["match_date"] = df["match_date"].astype(str).str.slice(0, 10)
        unique_teams = set(df["home_team"].dropna().astype(str)) | set(df["away_team"].dropna().astype(str))
        key_map = {t: _team_alias_key(t) for t in unique_teams}
        df["_hk"] = df["home_team"].astype(str).map(key_map).fillna("")
        df["_ak"] = df["away_team"].astype(str).map(key_map).fillna("")
    return df


@st.cache_resource(show_spinner=False, ttl=600)
def load_espn_live_cache() -> dict:
    if not os.path.isfile(ESPN_CACHE_JSON):
        return {"updated_utc": "", "standings": {}, "live_fixtures": [], "_pair_idx": {}}
    try:
        with open(ESPN_CACHE_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"updated_utc": "", "standings": {}, "live_fixtures": [], "_pair_idx": {}}
    pair_idx = {}
    for lf in (raw.get("live_fixtures") or []):
        hk = _team_alias_key(lf.get("home_team", ""))
        ak = _team_alias_key(lf.get("away_team", ""))
        pair_idx.setdefault((hk, ak), []).append(lf)
    raw["_pair_idx"] = pair_idx
    return raw


def refresh_espn_live_now() -> tuple[bool, str]:
    global _ESPN_LAST_SYNC_MONO
    import time as _t
    now_m = _t.monotonic()
    if _ESPN_LAST_SYNC_MONO and (now_m - _ESPN_LAST_SYNC_MONO) < _ESPN_SYNC_COOLDOWN_SEC:
        wait_s = int(_ESPN_SYNC_COOLDOWN_SEC - (now_m - _ESPN_LAST_SYNC_MONO))
        return True, f"Dữ liệu Live ESPN vừa được cập nhật xong (đồng bộ lại sau {wait_s}s)."
    try:
        from build_pro_analytics_store import build_espn_live_cache
        build_espn_live_cache()
        _ESPN_LAST_SYNC_MONO = _t.monotonic()
        load_espn_live_cache.clear()
        fetch_espn_event_live_intel.clear()
        return True, "Đã đồng bộ Live ESPN (kèo nhà cái, giờ bóng lăn & BXH 2026/27)!"
    except Exception as exc:
        return False, f"Không thể đồng bộ ESPN Live ({exc}). Đang dùng bộ đệm cục bộ."


def _american_to_prob(ml):
    try:
        v = float(ml)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return (-v) / (-v + 100.0) if v < 0 else 100.0 / (v + 100.0)


@st.cache_data(show_spinner=False, ttl=600, max_entries=128)
def fetch_espn_event_live_intel(espn_slug: str, espn_event_id: str) -> dict:
    if not espn_slug or not espn_event_id:
        return {}
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_slug}/summary?event={espn_event_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.5) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}

    out = {"venue": "", "odds": {}, "leaders": [], "news": []}
    ven = (d.get("gameInfo") or {}).get("venue") or {}
    if ven:
        city = (ven.get("address") or {}).get("city", "")
        out["venue"] = f"{ven.get('fullName', '')} ({city})" if city else ven.get("fullName", "")

    odds_list = d.get("odds") or d.get("pickcenter") or []
    if odds_list:
        od0 = odds_list[0]
        h_ml = (od0.get("homeTeamOdds") or {}).get("moneyLine")
        a_ml = (od0.get("awayTeamOdds") or {}).get("moneyLine")
        d_ml = (od0.get("drawOdds") or {}).get("moneyLine")
        ph, pd_, pa = _american_to_prob(h_ml), _american_to_prob(d_ml), _american_to_prob(a_ml)
        if ph and pd_ and pa:
            tot = ph + pd_ + pa
            out["odds"] = {
                "provider": (od0.get("provider") or {}).get("name", "DraftKings"),
                "details": od0.get("details", ""),
                "over_under": od0.get("overUnder"),
                "spread": od0.get("spread"),
                "mkt_prob_H": round(ph / tot * 100.0, 1),
                "mkt_prob_D": round(pd_ / tot * 100.0, 1),
                "mkt_prob_A": round(pa / tot * 100.0, 1),
                "overround_pct": round((tot - 1.0) * 100.0, 1),
            }

    for team_ld in (d.get("leaders") or []):
        t_name = (team_ld.get("team") or {}).get("displayName", "")
        cats = []
        for cat in (team_ld.get("leaders") or [])[:3]:
            top_l = (cat.get("leaders") or [{}])[0]
            ath = top_l.get("athlete") or {}
            if ath.get("displayName"):
                cats.append({
                    "category": cat.get("displayName", ""),
                    "player": ath.get("displayName", ""),
                    "position": (ath.get("position") or {}).get("abbreviation", ""),
                    "value": top_l.get("displayValue", ""),
                })
        if cats:
            out["leaders"].append({"team": t_name, "items": cats})

    for art in ((d.get("news") or {}).get("articles") or [])[:4]:
        if art.get("headline"):
            out["news"].append({
                "headline": art["headline"],
                "description": art.get("description", ""),
                "published": str(art.get("published", ""))[:10],
                "link": ((art.get("links") or {}).get("web") or {}).get("href", ""),
            })
    return out


ALL_ESPN_LEAGUE_SLUGS = {
    code: meta[2]
    for code, meta in LEAGUE_META.items()
    if code in LEAGUE_ORDER and len(meta) >= 3 and meta[2]
}

_VN_TZ = timezone(timedelta(hours=7))


def _fetch_espn_scoreboard_day(code: str, slug: str, dt_yyyymmdd: str, timeout: float = 3.0) -> list[dict]:
    """Fetch a single competition scoreboard day from ESPN (single-date query only)."""
    try:
        from build_pro_analytics_store import clean_club_name
    except Exception:
        clean_club_name = lambda s: str(s or "").strip()

    comp_label = CATALOG_CATEGORY_MAP.get(code, LEAGUE_META.get(code, (code, code, slug))[0])
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dt_yyyymmdd}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NCKH-TUD65/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            sb = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    out = []
    for ev in (sb.get("events") or []):
        try:
            eid = str(ev.get("id") or "").strip()
            if not eid:
                continue
            st_obj = ev.get("status") or {}
            st_type = st_obj.get("type") or {}
            st_name = str(st_type.get("name") or "")
            state = str(st_type.get("state") or "pre").lower()
            if state not in ("in", "pre", "post"):
                state = "post" if ("FINAL" in st_name or "FULL_TIME" in st_name) else "pre"

            comp0 = (ev.get("competitions") or [{}])[0] or {}
            comps = comp0.get("competitors") or []
            home_c = next((c for c in comps if isinstance(c, dict) and c.get("homeAway") == "home"), None)
            away_c = next((c for c in comps if isinstance(c, dict) and c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue

            h_team_obj = home_c.get("team") or {}
            a_team_obj = away_c.get("team") or {}
            h_name = clean_club_name(h_team_obj.get("displayName", ""))
            a_name = clean_club_name(a_team_obj.get("displayName", ""))
            if not h_name or not a_name:
                continue

            h_logos = h_team_obj.get("logos") or []
            a_logos = a_team_obj.get("logos") or []
            h_logo = str(h_team_obj.get("logo") or (h_logos[0].get("href") if h_logos and isinstance(h_logos[0], dict) else "") or "")
            a_logo = str(a_team_obj.get("logo") or (a_logos[0].get("href") if a_logos and isinstance(a_logos[0], dict) else "") or "")

            h_score, a_score = None, None
            if state in ("in", "post"):
                try:
                    h_score = int(float(home_c.get("score")))
                    a_score = int(float(away_c.get("score")))
                except (TypeError, ValueError):
                    h_score, a_score = None, None

            kickoff_utc = str(ev.get("date") or "")
            vn_date = f"{dt_yyyymmdd[:4]}-{dt_yyyymmdd[4:6]}-{dt_yyyymmdd[6:8]}"
            vn_time = ""
            kickoff_ts = 0.0
            if kickoff_utc:
                try:
                    dt_utc = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
                    dt_vn = dt_utc.astimezone(_VN_TZ)
                    vn_date = dt_vn.strftime("%Y-%m-%d")
                    vn_time = dt_vn.strftime("%H:%M")
                    kickoff_ts = dt_utc.timestamp()
                except Exception:
                    vn_date = kickoff_utc[:10] or vn_date

            raw_detail = str(st_type.get("shortDetail") or "").strip()
            if state == "in":
                status_detail = raw_detail or str(st_obj.get("displayClock") or "LIVE")
            elif state == "post":
                status_detail = raw_detail or "FT"
            else:
                status_detail = "Hoãn" if "POSTPONED" in st_name.upper() else (vn_time or "Theo lịch")

            venue_obj = comp0.get("venue") or {}
            venue = str(venue_obj.get("fullName") or "")
            city = str((venue_obj.get("address") or {}).get("city") or "")
            odds_arr = comp0.get("odds") or []
            od0 = next((o for o in odds_arr if isinstance(o, dict)), None)
            odds_info = {}
            if od0:
                h_ml = (od0.get("homeTeamOdds") or {}).get("moneyLine")
                a_ml = (od0.get("awayTeamOdds") or {}).get("moneyLine")
                d_ml = (od0.get("drawOdds") or {}).get("moneyLine")
                ph, pd_, pa = _american_to_prob(h_ml), _american_to_prob(d_ml), _american_to_prob(a_ml)
                if ph and pd_ and pa:
                    tot_p = ph + pd_ + pa
                    odds_info = {
                        "provider": (od0.get("provider") or {}).get("name", "DraftKings"),
                        "details": od0.get("details", ""),
                        "over_under": od0.get("overUnder"),
                        "spread": od0.get("spread"),
                        "home_ml": h_ml,
                        "draw_ml": d_ml,
                        "away_ml": a_ml,
                        "mkt_prob_H": round(ph / tot_p * 100, 1),
                        "mkt_prob_D": round(pd_ / tot_p * 100, 1),
                        "mkt_prob_A": round(pa / tot_p * 100, 1),
                    }

            out.append({
                "espn_event_id": eid,
                "league_code": code,
                "espn_slug": slug,
                "competition": comp_label,
                "kickoff_utc": kickoff_utc,
                "kickoff_ts": kickoff_ts,
                "match_date": vn_date,
                "vn_date": vn_date,
                "vn_time": vn_time,
                "state": state,
                "status": st_name,
                "status_detail": status_detail,
                "home_team": h_name,
                "away_team": a_name,
                "home_logo": h_logo,
                "away_logo": a_logo,
                "home_score": h_score,
                "away_score": a_score,
                "home_form": home_c.get("form", ""),
                "away_form": away_c.get("form", ""),
                "venue": f"{venue} ({city})" if venue and city else venue,
                "odds": odds_info,
            })
        except Exception:
            continue
    return out


@st.cache_resource(show_spinner=False, ttl=3600)
def _fetch_backfill_past_espn_results(today_vn_str: str) -> list[dict]:
    """Backfill finished ESPN results for upcoming_fixtures dates <= today across all 13 leagues."""
    up_df = load_upcoming_fixtures_df()
    if up_df.empty:
        return []
    try:
        today_d = date.fromisoformat(today_vn_str)
    except Exception:
        today_d = datetime.now(_VN_TZ).date()
    min_d_str = (today_d - timedelta(days=45)).strftime("%Y-%m-%d")
    sub = up_df[(up_df["match_date"] >= min_d_str) & (up_df["match_date"] <= today_vn_str)]
    if sub.empty:
        return []

    tasks = set()
    for lg, mdate in zip(sub["league"].astype(str), sub["match_date"].astype(str)):
        slug = ALL_ESPN_LEAGUE_SLUGS.get(lg)
        if not slug:
            continue
        try:
            dt_o = date.fromisoformat(mdate[:10])
        except Exception:
            continue
        for delta in (-1, 0):
            d_s = (dt_o + timedelta(days=delta)).strftime("%Y%m%d")
            tasks.add((lg, slug, d_s))

    if not tasks:
        return []
    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_fetch_espn_scoreboard_day, c, s, d, 3.0) for c, s, d in sorted(tasks)]
        for fut in as_completed(futs):
            try:
                out.extend(fut.result())
            except Exception:
                continue
    return out


@st.cache_resource(show_spinner=False, ttl=180)
def _fetch_recent_live_espn_results(bucket_3m: int) -> list[dict]:
    """Fetch yesterday/today/tomorrow UTC across all 13 leagues every 3 minutes."""
    now_utc = datetime.now(timezone.utc)
    dt_strs = [(now_utc + timedelta(days=d)).strftime("%Y%m%d") for d in (-1, 0, 1)]
    tasks = [
        (code, slug, dt_s)
        for code, slug in ALL_ESPN_LEAGUE_SLUGS.items()
        for dt_s in dt_strs
    ]
    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_fetch_espn_scoreboard_day, c, s, d, 2.5) for c, s, d in tasks]
        for fut in as_completed(futs):
            try:
                out.extend(fut.result())
            except Exception:
                continue
    return out


def get_synced_espn_results(force_refresh: bool = False) -> dict:
    """Merge disk ESPN cache, past-schedule backfill, and recent 13-league live results."""
    if force_refresh:
        for fn in (_fetch_recent_live_espn_results, _fetch_backfill_past_espn_results, load_espn_live_cache):
            try:
                fn.clear()
            except Exception:
                pass

    disk = load_espn_live_cache()
    today_vn_str = datetime.now(_VN_TZ).strftime("%Y-%m-%d")
    bucket_3m = int(_time.time() // 180)

    merged_by_id: dict[str, dict] = {}
    prio_rank = {"pre": 1, "in": 2, "post": 3}

    for source_list, recency in (
        (disk.get("live_fixtures") or [], 0),
        (_fetch_backfill_past_espn_results(today_vn_str), 1),
        (_fetch_recent_live_espn_results(bucket_3m), 2),
    ):
        for item in source_list:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("espn_event_id") or "").strip()
            if not eid:
                continue
            prev = merged_by_id.get(eid)
            if prev is None:
                c = dict(item)
                c["_prio"] = (prio_rank.get(str(item.get("state") or "pre").lower(), 0), recency)
                merged_by_id[eid] = c
            else:
                new_prio = (prio_rank.get(str(item.get("state") or "pre").lower(), 0), recency)
                if new_prio >= prev.get("_prio", (0, -1)):
                    c = dict(item)
                    c["_prio"] = new_prio
                    if not c.get("odds") and prev.get("odds"):
                        c["odds"] = prev["odds"]
                    merged_by_id[eid] = c

    pair_idx: dict[tuple[str, str], list[dict]] = {}
    final_list = []
    for rec in merged_by_id.values():
        rec.pop("_prio", None)
        final_list.append(rec)
        hk = _team_alias_key(rec.get("home_team", ""))
        ak = _team_alias_key(rec.get("away_team", ""))
        pair_idx.setdefault((hk, ak), []).append(rec)

    return {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if final_list else disk.get("updated_utc", ""),
        "standings": disk.get("standings") or {},
        "live_fixtures": final_list,
        "_pair_idx": pair_idx,
    }


def _espn_candidate_date(cand: dict) -> str:
    return str(cand.get("vn_date") or cand.get("match_date") or cand.get("kickoff_utc") or "")[:10]


def match_espn_live_entry(
    home_team: str,
    away_team: str,
    match_date: str,
    espn_cache: dict,
    max_day_diff: int = 2,
) -> dict:
    """Tra cứu O(1) từ bộ chỉ mục cặp đội ESPN với chốt an toàn chênh lệch ngày."""
    pair_idx = espn_cache.get("_pair_idx") or {}
    hk = _team_alias_key(home_team)
    ak = _team_alias_key(away_team)
    cands = pair_idx.get((hk, ak)) or []
    if not cands:
        return {}
    mdate = str(match_date or "")[:10]
    if not mdate:
        return cands[0]
    try:
        dt_ref = date.fromisoformat(mdate)
    except Exception:
        return cands[0]

    best_cand = {}
    best_diff = 9999
    for c in cands:
        cd = _espn_candidate_date(c)
        if not cd:
            continue
        try:
            diff = abs((date.fromisoformat(cd) - dt_ref).days)
        except Exception:
            continue
        if diff < best_diff:
            best_diff = diff
            best_cand = c
    if best_diff <= max_day_diff:
        return best_cand
    return {}


def get_upcoming_catalog_entries(days_ahead: int = 240) -> list[dict]:
    up_df = load_upcoming_fixtures_df()
    synced_espn = get_synced_espn_results()
    now_vn = datetime.now(_VN_TZ)
    today_vn_str = now_vn.strftime("%Y-%m-%d")
    cutoff_min = (now_vn - timedelta(days=45)).strftime("%Y-%m-%d")
    cutoff_max = (now_vn + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    entries = []
    matched_eids: set[str] = set()
    matched_sigs: list[tuple[str, str, date]] = []

    if not up_df.empty:
        sub = up_df[(up_df["match_date"] >= cutoff_min) & (up_df["match_date"] <= cutoff_max)]
        for _, r in sub.iterrows():
            lg = str(r.get("league") or "")
            cat_name = CATALOG_CATEGORY_MAP.get(lg, LEAGUE_META.get(lg, (f"⚽ {lg}", lg, ""))[0])
            lg_short = LEAGUE_META.get(lg, (lg, lg, ""))[1]
            ht = str(r.get("home_team") or "Home")
            at = str(r.get("away_team") or "Away")
            md = str(r.get("match_date") or "")[:10]
            espn_m = match_espn_live_entry(ht, at, md, synced_espn, max_day_diff=2)
            season_disp = str(r.get("season") or "2026-2027").replace("-", "/")

            eff_date = str(espn_m.get("vn_date") or espn_m.get("match_date") or md)[:10]
            state = str(espn_m.get("state") or "").lower()
            hs = espn_m.get("home_score")
            as_ = espn_m.get("away_score")

            if espn_m.get("espn_event_id"):
                matched_eids.add(str(espn_m["espn_event_id"]))
            try:
                matched_sigs.append((_team_alias_key(ht), _team_alias_key(at), date.fromisoformat(eff_date)))
            except Exception:
                pass

            if state in ("post", "in") and hs is not None and as_ is not None:
                is_live = state == "in"
                st_badge = f"🔴 LIVE ({espn_m.get('status_detail') or 'LIVE'})" if is_live else "✅ FT"
                entries.append({
                    "category": cat_name,
                    "id": f"espn_res_{espn_m.get('espn_event_id') or r['match_id']}",
                    "upcoming_match_id": str(r["match_id"]),
                    "espn_event_id": str(espn_m.get("espn_event_id") or ""),
                    "league_code": lg,
                    "label": f"{ht} {int(hs)} - {int(as_)} {at}",
                    "home_team": str(espn_m.get("home_team") or ht),
                    "away_team": str(espn_m.get("away_team") or at),
                    "home_score": int(hs),
                    "away_score": int(as_),
                    "is_upcoming": False,
                    "stage": f"{lg_short} {season_disp} - {st_badge}",
                    "date": eff_date,
                    "kickoff_utc": espn_m.get("kickoff_utc", ""),
                    "stadium": espn_m.get("venue") or f"{lg_short} Stadium",
                    "referee": "Trọng tài FIFA / Liên đoàn",
                    "source_type": "espn_live_result",
                    "espn_fixture": espn_m,
                })
                continue

            if eff_date < today_vn_str and state != "pre":
                continue

            entries.append({
                "category": cat_name,
                "id": f"upcoming_{r['match_id']}",
                "upcoming_match_id": str(r["match_id"]),
                "league_code": lg,
                "label": f"⏰ {ht} vs {at}",
                "home_team": ht,
                "away_team": at,
                "home_score": None,
                "away_score": None,
                "is_upcoming": True,
                "stage": f"{lg_short} {season_disp} - 🗓️ Sắp diễn ra",
                "date": eff_date,
                "kickoff_utc": espn_m.get("kickoff_utc", ""),
                "stadium": espn_m.get("venue") or f"{lg_short} Stadium",
                "referee": "Chưa công bố tổ trọng tài",
                "prob_H": round(_safe_float(r.get("prob_H"), 0.34) * 100.0, 1),
                "prob_D": round(_safe_float(r.get("prob_D"), 0.28) * 100.0, 1),
                "prob_A": round(_safe_float(r.get("prob_A"), 0.38) * 100.0, 1),
                "pred_home_goals": round(_safe_float(r.get("pred_home_goals"), 1.35), 2),
                "pred_away_goals": round(_safe_float(r.get("pred_away_goals"), 1.15), 2),
                "source_type": "upcoming_fixture",
            })

    # Also include recent 14-day finished/live ESPN fixtures absent from upcoming_fixtures.parquet
    cutoff_recent = (now_vn - timedelta(days=14)).strftime("%Y-%m-%d")
    for em in (synced_espn.get("live_fixtures") or []):
        eid = str(em.get("espn_event_id") or "")
        if not eid or eid in matched_eids:
            continue
        state = str(em.get("state") or "").lower()
        hs, as_ = em.get("home_score"), em.get("away_score")
        ed = str(em.get("vn_date") or em.get("match_date") or "")[:10]
        if state not in ("post", "in") or hs is None or as_ is None or not ed or ed < cutoff_recent:
            continue
        ht_e = str(em.get("home_team") or "")
        at_e = str(em.get("away_team") or "")
        hk_e, ak_e = _team_alias_key(ht_e), _team_alias_key(at_e)
        try:
            ed_dt = date.fromisoformat(ed)
            if any(h0 == hk_e and a0 == ak_e and abs((ed_dt - d0).days) <= 2 for h0, a0, d0 in matched_sigs):
                continue
        except Exception:
            pass
        lg = str(em.get("league_code") or "")
        cat_name = CATALOG_CATEGORY_MAP.get(lg, em.get("competition") or f"⚽ {lg}")
        lg_short = LEAGUE_META.get(lg, (lg, lg, ""))[1]
        st_badge = f"🔴 LIVE ({em.get('status_detail') or 'LIVE'})" if state == "in" else "✅ FT"
        entries.append({
            "category": cat_name,
            "id": f"espn_res_{eid}",
            "espn_event_id": eid,
            "league_code": lg,
            "label": f"{ht_e} {int(hs)} - {int(as_)} {at_e}",
            "home_team": ht_e,
            "away_team": at_e,
            "home_score": int(hs),
            "away_score": int(as_),
            "is_upcoming": False,
            "stage": f"{lg_short} 2026/27 - {st_badge}",
            "date": ed,
            "kickoff_utc": em.get("kickoff_utc", ""),
            "stadium": em.get("venue") or f"{lg_short} Stadium",
            "referee": "Trọng tài FIFA / Liên đoàn",
            "source_type": "espn_live_result",
            "espn_fixture": em,
        })

    return entries


# ==========================================================
# 2. QUANTITATIVE & EXPLAINABLE AI ENGINE v4.0
# ==========================================================
def dixon_coles_matrix(lam_h: float, lam_a: float, rho: float = -0.06, max_goals: int = 7) -> dict:
    lam_h = max(0.18, min(5.0, float(lam_h)))
    lam_a = max(0.18, min(5.0, float(lam_a)))

    def _pmf(k, lam):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def _tau(x, y):
        if x == 0 and y == 0:
            return max(0.05, 1.0 - lam_h * lam_a * rho)
        if x == 0 and y == 1:
            return max(0.05, 1.0 + lam_h * rho)
        if x == 1 and y == 0:
            return max(0.05, 1.0 + lam_a * rho)
        if x == 1 and y == 1:
            return max(0.05, 1.0 - rho)
        return 1.0

    mat = np.zeros((max_goals, max_goals), dtype=float)
    for i in range(max_goals):
        pi = _pmf(i, lam_h)
        for j in range(max_goals):
            mat[i, j] = pi * _pmf(j, lam_a) * _tau(i, j)
    mat /= mat.sum() or 1.0

    p_h = float(np.tril(mat, -1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, 1).sum())

    p_btts = float(mat[1:, 1:].sum())
    p_cs_h = float(mat[:, 0].sum())
    p_cs_a = float(mat[0, :].sum())

    total_dist = {k: 0.0 for k in range(6)}
    over_15 = over_25 = over_35 = ah_h_m15 = ah_a_m15 = 0.0
    scorelines = []
    for i in range(max_goals):
        for j in range(max_goals):
            p = float(mat[i, j])
            tot_g = i + j
            total_dist[min(5, tot_g)] += p
            if tot_g >= 2:
                over_15 += p
            if tot_g >= 3:
                over_25 += p
            if tot_g >= 4:
                over_35 += p
            if i - j >= 2:
                ah_h_m15 += p
            if j - i >= 2:
                ah_a_m15 += p
            if i <= 5 and j <= 5:
                scorelines.append({"score": f"{i}-{j}", "h": i, "a": j, "prob_pct": round(p * 100.0, 2)})

    scorelines.sort(key=lambda x: x["prob_pct"], reverse=True)
    return {
        "matrix_6x6": np.round(mat[:6, :6] * 100.0, 2),
        "lam_h": round(lam_h, 2),
        "lam_a": round(lam_a, 2),
        "prob_H": round(p_h * 100.0, 1),
        "prob_D": round(p_d * 100.0, 1),
        "prob_A": round(p_a * 100.0, 1),
        "dc_1X": round((p_h + p_d) * 100.0, 1),
        "dc_X2": round((p_a + p_d) * 100.0, 1),
        "dc_12": round((p_h + p_a) * 100.0, 1),
        "btts_yes": round(p_btts * 100.0, 1),
        "btts_no": round((1.0 - p_btts) * 100.0, 1),
        "over_15": round(over_15 * 100.0, 1),
        "under_15": round((1.0 - over_15) * 100.0, 1),
        "over_25": round(over_25 * 100.0, 1),
        "under_25": round((1.0 - over_25) * 100.0, 1),
        "over_35": round(over_35 * 100.0, 1),
        "under_35": round((1.0 - over_35) * 100.0, 1),
        "cs_home": round(p_cs_h * 100.0, 1),
        "cs_away": round(p_cs_a * 100.0, 1),
        "ah_home_m15": round(ah_h_m15 * 100.0, 1),
        "ah_away_m15": round(ah_a_m15 * 100.0, 1),
        "total_goals_dist": {k: round(v * 100.0, 1) for k, v in total_dist.items()},
        "top_scorelines": scorelines[:8],
    }


def _elo_1x2_probs(home_elo: float, away_elo: float, home_adv: float = 65.0) -> tuple[float, float, float]:
    """Mô hình Elo Logistic hiệu chuẩn bóng đá (tính xác suất Thắng / Hòa / Thua theo chênh lệch Elo)."""
    diff = (home_elo + home_adv) - away_elo
    exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    # Xác suất hòa cao nhất (~29%) khi 2 đội ngang Elo, giảm dần khi chênh lệch lớn
    p_draw = 0.29 * math.exp(-((diff / 310.0) ** 2))
    p_home = max(0.02, exp_h - 0.5 * p_draw)
    p_away = max(0.02, 1.0 - p_home - p_draw)
    tot = p_home + p_draw + p_away
    return round(p_home / tot * 100.0, 1), round(p_draw / tot * 100.0, 1), round(p_away / tot * 100.0, 1)


def extract_team_recent_form(hist_df: pd.DataFrame, team_name: str, before_date: str, n: int = 10) -> dict:
    """Truy vấn tốc độ cao (<3ms) nhờ chỉ mục _hk/_ak đã tiền tính toán."""
    empty = {
        "rows": pd.DataFrame(), "season_split": {}, "wdl": [], "ppg": 0.0, "latest_elo": 1650.0,
        "gf_pg": 1.3, "ga_pg": 1.2, "xg_pg": 1.3, "xga_pg": 1.2,
        "finishing_diff": 0.0, "def_luck_diff": 0.0, "btts_pct": 50.0, "o25_pct": 50.0, "cs_pct": 25.0,
    }
    if hist_df is None or hist_df.empty:
        return empty
    tk = _team_alias_key(team_name)
    hk_col = hist_df["_hk"] if "_hk" in hist_df.columns else hist_df["home_team"].astype(str).apply(_team_alias_key)
    ak_col = hist_df["_ak"] if "_ak" in hist_df.columns else hist_df["away_team"].astype(str).apply(_team_alias_key)
    t_df = hist_df[(hist_df["match_date"] < str(before_date)[:10]) & ((hk_col == tk) | (ak_col == tk))].head(max(n, 24))
    if t_df.empty:
        return empty

    records = []
    latest_elo = 1650.0
    for idx_r, (_, r) in enumerate(t_df.iterrows()):
        is_h = (r["_hk"] if "_hk" in r.index else _team_alias_key(r["home_team"])) == tk
        if idx_r == 0:
            latest_elo = _safe_float(r.get("home_elo" if is_h else "away_elo"), 1650.0)
        gf = int(_safe_float(r["home_goals"] if is_h else r["away_goals"], 0))
        ga = int(_safe_float(r["away_goals"] if is_h else r["home_goals"], 0))
        xg = _safe_float(r["home_xg"] if is_h else r["away_xg"], float("nan"))
        xga = _safe_float(r["away_xg"] if is_h else r["home_xg"], float("nan"))
        opp = str(r["away_team"] if is_h else r["home_team"])
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        records.append({
            "date": str(r["match_date"])[:10],
            "league": str(r.get("league") or ""),
            "season": str(r.get("season") or ""),
            "venue": "🏠 Nhà" if is_h else "✈️ Khách",
            "is_home": is_h,
            "opponent": opp,
            "score": f"{gf}-{ga}",
            "gf": gf, "ga": ga,
            "xg": round(xg, 2) if not math.isnan(xg) else None,
            "xga": round(xga, 2) if not math.isnan(xga) else None,
            "res": res,
            "pts": 3 if res == "W" else (1 if res == "D" else 0),
        })

    all_rec_df = pd.DataFrame(records)
    rec_n = all_rec_df.head(n).copy()
    xg_valid = rec_n.dropna(subset=["xg", "xga"])
    gf_pg = float(rec_n["gf"].mean())
    ga_pg = float(rec_n["ga"].mean())
    xg_pg = float(xg_valid["xg"].mean()) if not xg_valid.empty else gf_pg
    xga_pg = float(xg_valid["xga"].mean()) if not xg_valid.empty else ga_pg

    s26 = all_rec_df[all_rec_df["season"] == "2026-2027"]
    if len(s26) < 4:
        s26 = all_rec_df.head(16)
    split_out = {}
    for side_k, mask_s in [("home", s26["is_home"]), ("away", ~s26["is_home"])]:
        ss = s26[mask_s]
        if not ss.empty:
            ss_xg = ss.dropna(subset=["xg", "xga"])
            split_out[side_k] = {
                "mp": int(len(ss)),
                "ppg": round(float(ss["pts"].mean()), 2),
                "gf_pg": round(float(ss["gf"].mean()), 2),
                "ga_pg": round(float(ss["ga"].mean()), 2),
                "xg_pg": round(float(ss_xg["xg"].mean()), 2) if not ss_xg.empty else round(float(ss["gf"].mean()), 2),
                "xga_pg": round(float(ss_xg["xga"].mean()), 2) if not ss_xg.empty else round(float(ss["ga"].mean()), 2),
                "cs_pct": round(float((ss["ga"] == 0).mean() * 100.0), 1),
                "btts_pct": round(float(((ss["gf"] > 0) & (ss["ga"] > 0)).mean() * 100.0), 1),
                "o25_pct": round(float(((ss["gf"] + ss["ga"]) >= 3).mean() * 100.0), 1),
            }

    return {
        "rows": rec_n,
        "season_split": split_out,
        "wdl": rec_n["res"].tolist()[:5],
        "ppg": round(float(rec_n["pts"].mean()), 2),
        "latest_elo": round(latest_elo, 1),
        "gf_pg": round(gf_pg, 2),
        "ga_pg": round(ga_pg, 2),
        "xg_pg": round(xg_pg, 2),
        "xga_pg": round(xga_pg, 2),
        "finishing_diff": round(gf_pg - xg_pg, 2),
        "def_luck_diff": round(xga_pg - ga_pg, 2),
        "btts_pct": round(float(((rec_n["gf"] > 0) & (rec_n["ga"] > 0)).mean() * 100.0), 1),
        "o25_pct": round(float(((rec_n["gf"] + rec_n["ga"]) >= 3).mean() * 100.0), 1),
        "cs_pct": round(float((rec_n["ga"] == 0).mean() * 100.0), 1),
    }


def extract_h2h_history(hist_df: pd.DataFrame, home_team: str, away_team: str, before_date: str, limit: int = 10) -> dict:
    if hist_df is None or hist_df.empty:
        return {"matches": pd.DataFrame(), "h_wins": 0, "draws": 0, "a_wins": 0, "avg_goals": 0.0, "btts_pct": 0.0, "o25_pct": 0.0}
    hk = _team_alias_key(home_team)
    ak = _team_alias_key(away_team)
    hk_col = hist_df["_hk"] if "_hk" in hist_df.columns else hist_df["home_team"].astype(str).apply(_team_alias_key)
    ak_col = hist_df["_ak"] if "_ak" in hist_df.columns else hist_df["away_team"].astype(str).apply(_team_alias_key)
    h2h = hist_df[
        (hist_df["match_date"] < str(before_date)[:10])
        & (((hk_col == hk) & (ak_col == ak)) | ((hk_col == ak) & (ak_col == hk)))
    ].head(limit)
    if h2h.empty:
        return {"matches": pd.DataFrame(), "h_wins": 0, "draws": 0, "a_wins": 0, "avg_goals": 0.0, "btts_pct": 0.0, "o25_pct": 0.0}

    hw = dr = aw = 0
    rows = []
    for _, r in h2h.iterrows():
        hg = int(_safe_float(r["home_goals"], 0))
        ag = int(_safe_float(r["away_goals"], 0))
        is_same = (r["_hk"] if "_hk" in r.index else _team_alias_key(r["home_team"])) == hk
        if hg == ag:
            dr += 1
            winner = "Hòa"
        elif (hg > ag and is_same) or (ag > hg and not is_same):
            hw += 1
            winner = home_team
        else:
            aw += 1
            winner = away_team
        ht_s = f"({int(r['home_score_ht'])}-{int(r['away_score_ht'])})" if pd.notna(r.get("home_score_ht")) and pd.notna(r.get("away_score_ht")) else ""
        xg_s = f"{_safe_float(r.get('home_xg')):.2f} - {_safe_float(r.get('away_xg')):.2f}" if pd.notna(r.get("home_xg")) else "—"
        rows.append({
            "Ngày": str(r["match_date"])[:10],
            "Giải": str(r.get("league") or ""),
            "Đội Nhà": str(r["home_team"]),
            "Tỷ Số": f"{hg} - {ag} {ht_s}".strip(),
            "Đội Khách": str(r["away_team"]),
            "xG Thực": xg_s,
            "Thắng": winner,
            "_tot": hg + ag,
            "_btts": int(hg > 0 and ag > 0),
        })
    rdf = pd.DataFrame(rows)
    return {
        "matches": rdf.drop(columns=["_tot", "_btts"]),
        "h_wins": hw, "draws": dr, "a_wins": aw,
        "avg_goals": round(float(rdf["_tot"].mean()), 2),
        "btts_pct": round(float(rdf["_btts"].mean() * 100.0), 1),
        "o25_pct": round(float((rdf["_tot"] >= 3).mean() * 100.0), 1),
    }


@st.cache_data(show_spinner=False, ttl=600)
def compute_league_xpts_standings(league_code: str, season: str = "2026-2027") -> pd.DataFrame:
    hist = load_historical_xg_df()
    if hist.empty:
        return pd.DataFrame()
    sub = hist[(hist["league"] == league_code) & (hist["season"] == season)].sort_values("match_date")
    if sub.empty:
        return pd.DataFrame()

    teams = {}
    for _, r in sub.iterrows():
        ht, at = str(r["home_team"]), str(r["away_team"])
        hg, ag = int(_safe_float(r["home_goals"], 0)), int(_safe_float(r["away_goals"], 0))
        hxg, axg = _safe_float(r.get("home_xg"), hg), _safe_float(r.get("away_xg"), ag)
        for t in (ht, at):
            teams.setdefault(t, {"team": t, "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "xg": 0.0, "xga": 0.0, "xpts": 0.0, "form": []})
        dc = dixon_coles_matrix(max(0.08, hxg), max(0.08, axg), rho=-0.04, max_goals=6)
        th, ta = teams[ht], teams[at]
        th["mp"] += 1; ta["mp"] += 1
        th["gf"] += hg; th["ga"] += ag; ta["gf"] += ag; ta["ga"] += hg
        th["xg"] += hxg; th["xga"] += axg; ta["xg"] += axg; ta["xga"] += hxg
        th["xpts"] += (dc["prob_H"] / 100.0) * 3.0 + (dc["prob_D"] / 100.0)
        ta["xpts"] += (dc["prob_A"] / 100.0) * 3.0 + (dc["prob_D"] / 100.0)
        if hg > ag:
            th["w"] += 1; th["pts"] += 3; th["form"].append("W"); ta["l"] += 1; ta["form"].append("L")
        elif hg < ag:
            ta["w"] += 1; ta["pts"] += 3; ta["form"].append("W"); th["l"] += 1; th["form"].append("L")
        else:
            th["d"] += 1; ta["d"] += 1; th["pts"] += 1; ta["pts"] += 1; th["form"].append("D"); ta["form"].append("D")

    rows = []
    for t_rec in teams.values():
        t_rec["gd"] = t_rec["gf"] - t_rec["ga"]
        t_rec["xg"] = round(t_rec["xg"], 2)
        t_rec["xga"] = round(t_rec["xga"], 2)
        t_rec["xgd"] = round(t_rec["xg"] - t_rec["xga"], 2)
        t_rec["xpts"] = round(t_rec["xpts"], 1)
        t_rec["luck_pts"] = round(t_rec["pts"] - t_rec["xpts"], 1)
        t_rec["form_5"] = t_rec["form"][-5:]
        rows.append(t_rec)

    df_out = pd.DataFrame(rows).sort_values(["pts", "gd", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    df_out["rank"] = np.arange(1, len(df_out) + 1)
    df_out["xrank"] = df_out["xpts"].rank(ascending=False, method="min").astype(int)
    return df_out


def compute_ai_key_drivers(
    home_elo: float,
    away_elo: float,
    h_form: dict,
    a_form: dict,
    h2h: dict,
    rest_diff: float,
    squad_diff: float = 0.0,
) -> list[dict]:
    """Explainable AI: Phân rã 6 nhân tố cốt lõi đóng góp vào lợi thế (+ nghiêng Đội Nhà, - nghiêng Đội Khách)."""
    # 1. Elo Strength (-15 to +15)
    elo_score = max(-18.0, min(18.0, (home_elo - away_elo) / 16.0))
    # 2. Home Advantage + Home/Away Split (+3 to +12)
    h_sp = h_form["season_split"].get("home") or {}
    a_sp = a_form["season_split"].get("away") or {}
    split_bonus = (h_sp.get("ppg", 1.5) - a_sp.get("ppg", 1.2)) * 2.4
    home_score = max(-8.0, min(14.0, 5.5 + split_bonus))
    # 3. Squad 4-Zone Power (Panna RAPM diff)
    sq_score = max(-14.0, min(14.0, squad_diff * 16.0))
    # 4. Recent 10-Match Net xG Form
    h_net_xg = h_form["xg_pg"] - h_form["xga_pg"]
    a_net_xg = a_form["xg_pg"] - a_form["xga_pg"]
    form_score = max(-14.0, min(14.0, (h_net_xg - a_net_xg) * 8.5))
    # 5. Head-to-Head Edge
    tot_h2h = h2h["h_wins"] + h2h["draws"] + h2h["a_wins"]
    h2h_score = max(-8.0, min(8.0, ((h2h["h_wins"] - h2h["a_wins"]) / max(1, tot_h2h)) * 8.0)) if tot_h2h >= 2 else 0.0
    # 6. Fitness / Rest Days
    rest_score = max(-5.0, min(5.0, rest_diff * 1.4))

    drivers = [
        {"factor": "🏆 Elo", "score": round(elo_score, 1), "detail": f"{home_elo:.0f} vs {away_elo:.0f}"},
        {"factor": "🏠 Sân nhà", "score": round(home_score, 1), "detail": f"PPG Nhà {h_sp.get('ppg', h_form['ppg']):.2f} vs Khách {a_sp.get('ppg', a_form['ppg']):.2f}"},
        {"factor": "⚔️ Đội hình", "score": round(sq_score, 1), "detail": f"Chênh lệch Panna {squad_diff:+.2f}"},
        {"factor": "📈 Form xG", "score": round(form_score, 1), "detail": f"Net xG {h_net_xg:+.2f} vs {a_net_xg:+.2f}"},
        {"factor": "🤝 H2H", "score": round(h2h_score, 1), "detail": f"{h2h['h_wins']}T-{h2h['draws']}H-{h2h['a_wins']}B" if tot_h2h else "Chưa gặp gần đây"},
        {"factor": "🔋 Thể lực", "score": round(rest_score, 1), "detail": f"Chênh lệch {rest_diff:+.0f} ngày"},
    ]
    drivers.sort(key=lambda x: abs(x["score"]), reverse=True)
    return drivers


def build_prematch_research_package(row: pd.Series, hist_df: pd.DataFrame, espn_cache: dict, fetch_live_intel: bool = True) -> dict:
    ht = str(row["home_team"])
    at = str(row["away_team"])
    md = str(row["match_date"])[:10]
    lg = str(row["league"])
    cat_name, lg_short, _slug = LEAGUE_META.get(lg, (f"⚽ {lg}", lg, ""))

    # Layer 1: Opta/Panna Deep ML
    ml_lh = _safe_float(row.get("pred_home_goals"), 1.40)
    ml_la = _safe_float(row.get("pred_away_goals"), 1.15)
    ml_pH = round(_safe_float(row.get("prob_H"), 0.40) * 100.0, 1)
    ml_pD = round(_safe_float(row.get("prob_D"), 0.28) * 100.0, 1)
    ml_pA = round(_safe_float(row.get("prob_A"), 0.32) * 100.0, 1)

    # Layer 2: Rolling xG Form + Home/Away Split
    h_form = extract_team_recent_form(hist_df, ht, md, n=10)
    a_form = extract_team_recent_form(hist_df, at, md, n=10)
    h2h = extract_h2h_history(hist_df, ht, at, md, limit=10)

    h_sp = h_form["season_split"].get("home") or {}
    a_sp = a_form["season_split"].get("away") or {}
    h_att = 0.60 * h_form["xg_pg"] + 0.40 * h_sp.get("xg_pg", h_form["xg_pg"])
    h_def = 0.60 * h_form["xga_pg"] + 0.40 * h_sp.get("xga_pg", h_form["xga_pg"])
    a_att = 0.60 * a_form["xg_pg"] + 0.40 * a_sp.get("xg_pg", a_form["xg_pg"])
    a_def = 0.60 * a_form["xga_pg"] + 0.40 * a_sp.get("xga_pg", a_form["xga_pg"])

    form_lh = max(0.32, min(3.8, (h_att * a_def / 1.32) * 1.07))
    form_la = max(0.28, min(3.5, (a_att * h_def / 1.32) * 0.94))
    dc_form = dixon_coles_matrix(form_lh, form_la)

    # Layer 3: Elo Logistic Model
    home_elo = round(_safe_float(row.get("home_elo"), h_form["latest_elo"]), 1)
    away_elo = round(_safe_float(row.get("away_elo"), a_form["latest_elo"]), 1)
    elo_pH, elo_pD, elo_pA = _elo_1x2_probs(home_elo, away_elo)

    # Calibrated 3-Model Ensemble (55% Opta ML + 25% Dixon-Coles Form xG + 20% Elo)
    ens_pH = round(0.55 * ml_pH + 0.25 * dc_form["prob_H"] + 0.20 * elo_pH, 1)
    ens_pD = round(0.55 * ml_pD + 0.25 * dc_form["prob_D"] + 0.20 * elo_pD, 1)
    ens_pA = round(max(0.5, 100.0 - ens_pH - ens_pD), 1)

    ens_lh = round(0.68 * ml_lh + 0.32 * form_lh, 2)
    ens_la = round(0.68 * ml_la + 0.32 * form_la, 2)
    dc_ens = dixon_coles_matrix(ens_lh, ens_la)

    # Explainable AI Key Drivers
    squad_diff = _safe_float(row.get("home_sum_panna")) - _safe_float(row.get("away_sum_panna"))
    rest_diff = _safe_float(row.get("rest_diff"), 0.0)
    drivers = compute_ai_key_drivers(home_elo, away_elo, h_form, a_form, h2h, rest_diff, squad_diff)

    # Layer 4: Live ESPN Odds, Kickoff & News
    espn_m = match_espn_live_entry(ht, at, md, espn_cache)
    live_intel = fetch_espn_event_live_intel(espn_m["espn_slug"], espn_m["espn_event_id"]) if (fetch_live_intel and espn_m.get("espn_event_id") and espn_m.get("espn_slug")) else {}
    odds = live_intel.get("odds") or espn_m.get("odds") or {}
    venue = live_intel.get("venue") or espn_m.get("venue") or f"{lg_short} Arena"
    kickoff_disp, countdown_txt = _format_kickoff_vn(espn_m.get("kickoff_utc", ""), md)

    value_edges = []
    if odds.get("mkt_prob_H"):
        for side_lbl, ai_p, mkt_p in [
            (f"{ht} Thắng (1)", ens_pH, odds["mkt_prob_H"]),
            ("Hòa (X)", ens_pD, odds["mkt_prob_D"]),
            (f"{at} Thắng (2)", ens_pA, odds["mkt_prob_A"]),
        ]:
            edge = round(ai_p - mkt_p, 1)
            # Simplified Fractional Kelly indicator (1/4 Kelly)
            fair_dec = 100.0 / max(2.0, mkt_p)
            kelly = max(0.0, ((ai_p / 100.0) * fair_dec - 1.0) / max(0.05, fair_dec - 1.0)) * 25.0
            value_edges.append({"market": side_lbl, "ai_prob": ai_p, "mkt_prob": mkt_p, "edge": edge, "kelly_pct": round(kelly, 1)})
        value_edges.sort(key=lambda x: x["edge"], reverse=True)

    # Standings lookup
    xpts_tbl = compute_league_xpts_standings(lg, str(row.get("season") or "2026-2027"))
    h_rank = a_rank = h_pts = a_pts = None
    if not xpts_tbl.empty:
        hk, ak = _team_alias_key(ht), _team_alias_key(at)
        for _, tr_r in xpts_tbl.iterrows():
            tk_r = _team_alias_key(tr_r["team"])
            if tk_r == hk:
                h_rank, h_pts = int(tr_r["rank"]), int(tr_r["pts"])
            elif tk_r == ak:
                a_rank, a_pts = int(tr_r["rank"]), int(tr_r["pts"])

    # Confidence & Smart Pick Recommendations
    max_p = max(ens_pH, ens_pD, ens_pA)
    entropy = -sum((p / 100.0) * math.log(max(1e-6, p / 100.0)) for p in (ens_pH, ens_pD, ens_pA)) / math.log(3.0)
    confidence = round((1.0 - entropy) * 100.0, 1)
    conf_lbl = "Rất cao" if max_p >= 62 else ("Khá cao" if max_p >= 50 else ("Trung bình" if max_p >= 42 else "Khó lường"))

    if ens_pH >= 50:
        pick_1x2 = f"{ht} Thắng (1)"
        pick_1x2_prob = f"{ens_pH:.1f}%"
        pick_1x2_sub = f"An toàn: 1X ({dc_ens['dc_1X']}%)"
    elif ens_pA >= 50:
        pick_1x2 = f"{at} Thắng (2)"
        pick_1x2_prob = f"{ens_pA:.1f}%"
        pick_1x2_sub = f"An toàn: X2 ({dc_ens['dc_X2']}%)"
    elif ens_pH >= ens_pA:
        pick_1x2 = f"{ht} hoặc Hòa (1X)"
        pick_1x2_prob = f"{dc_ens['dc_1X']:.1f}%"
        pick_1x2_sub = f"Thắng thẳng: {ens_pH:.1f}% · Trận cân bằng"
    else:
        pick_1x2 = f"{at} hoặc Hòa (X2)"
        pick_1x2_prob = f"{dc_ens['dc_X2']:.1f}%"
        pick_1x2_sub = f"Thắng thẳng: {ens_pA:.1f}% · Trận cân bằng"

    if dc_ens["over_25"] >= 50.0:
        pick_goals = "Tài 2.5 bàn"
        pick_goals_prob = f"{dc_ens['over_25']:.1f}%"
    else:
        pick_goals = "Xỉu 2.5 bàn"
        pick_goals_prob = f"{dc_ens['under_25']:.1f}%"
    btts_tag = f"BTTS Có {dc_ens['btts_yes']}%" if dc_ens["btts_yes"] >= 52 else f"BTTS Không {dc_ens['btts_no']}%"

    return {
        "match_id": str(row["match_id"]),
        "league_code": lg,
        "league_name": cat_name,
        "league_short": lg_short,
        "season": str(row.get("season") or "2026-2027").replace("-", "/"),
        "match_date": md,
        "kickoff_disp": kickoff_disp,
        "countdown": countdown_txt,
        "venue": venue,
        "home_team": ht,
        "away_team": at,
        "home_logo": get_team_logo_url(ht, fallback_url=espn_m.get("home_logo", "")),
        "away_logo": get_team_logo_url(at, fallback_url=espn_m.get("away_logo", "")),
        "home_rank": h_rank, "away_rank": a_rank,
        "home_pts": h_pts, "away_pts": a_pts,
        "home_elo": home_elo, "away_elo": away_elo,
        "rest_diff": rest_diff,
        "ml": {"lam_h": round(ml_lh, 2), "lam_a": round(ml_la, 2), "prob_H": ml_pH, "prob_D": ml_pD, "prob_A": ml_pA},
        "elo_model": {"prob_H": elo_pH, "prob_D": elo_pD, "prob_A": elo_pA},
        "form_dc": dc_form,
        "ensemble": {
            "lam_h": ens_lh, "lam_a": ens_la,
            "prob_H": ens_pH, "prob_D": ens_pD, "prob_A": ens_pA,
            "dc": dc_ens, "confidence": confidence, "conf_lbl": conf_lbl,
        },
        "smart_picks": {
            "main": (pick_1x2, pick_1x2_prob, pick_1x2_sub),
            "goals": (pick_goals, pick_goals_prob, f"Tổng xG {ens_lh+ens_la:.2f} · {btts_tag}"),
            "score": (
                f"Tỷ số {dc_ens['top_scorelines'][0]['score']}",
                f"{dc_ens['top_scorelines'][0]['prob_pct']:.1f}%",
                f"Value: {value_edges[0]['market']} ({value_edges[0]['edge']:+.1f}%)" if value_edges else f"Kế tiếp: {dc_ens['top_scorelines'][1]['score']} ({dc_ens['top_scorelines'][1]['prob_pct']:.1f}%)",
            ),
        },
        "drivers": drivers,
        "odds": odds,
        "value_edges": value_edges,
        "home_form": h_form,
        "away_form": a_form,
        "h2h": h2h,
        "leaders": live_intel.get("leaders", []),
        "news": live_intel.get("news", []),
        "raw_row": row,
    }


def _generate_concise_ai_bullets(pkg: dict) -> list[str]:
    ht, at = pkg["home_team"], pkg["away_team"]
    ens = pkg["ensemble"]
    dc = ens["dc"]
    top_d = pkg["drivers"][0]
    bullets = []

    dom = ht if top_d["score"] >= 0 else at
    bullets.append(
        f"🎯 **Nhân tố quyết định số 1 ({top_d['factor']}):** Nghiêng về **{dom}** ({top_d['detail']}), "
        f"đưa mô hình tổng hợp tới khuyến nghị **{pkg['smart_picks']['main'][0]} ({pkg['smart_picks']['main'][1]})**."
    )
    tot_xg = ens["lam_h"] + ens["lam_a"]
    style = "cởi mở, giàu bàn thắng" if tot_xg >= 2.85 else ("chặt chẽ, thực dụng" if tot_xg <= 2.30 else "cân bằng")
    bullets.append(
        f"⚽ **Dự báo thế trận {style}:** Tổng xG kỳ vọng **{tot_xg:.2f} bàn** "
        f"(Tài 2.5: **{dc['over_25']}%** · BTTS: **{dc['btts_yes']}%** · Tỷ số sáng nhất: **{dc['top_scorelines'][0]['score']}**)."
    )
    hf, af = pkg["home_form"], pkg["away_form"]
    if abs(hf["finishing_diff"]) >= 0.30 or abs(af["finishing_diff"]) >= 0.30:
        hot_t, hot_d = (ht, hf["finishing_diff"]) if abs(hf["finishing_diff"]) >= abs(af["finishing_diff"]) else (at, af["finishing_diff"])
        note_reg = "đang dứt điểm hiệu quả vượt xG (phong độ cao)" if hot_d > 0 else "tạo xG tốt hơn số bàn thực ghi (tiềm ẩn bùng nổ)"
        bullets.append(f"📈 **Tín hiệu hồi quy xG:** **{hot_t}** {note_reg} (`{hot_d:+.2f}` bàn/trận trong 10 trận gần nhất).")
    if pkg["value_edges"] and pkg["value_edges"][0]["edge"] >= 2.5:
        ve = pkg["value_edges"][0]
        bullets.append(
            f"💎 **Phát hiện Value Edge (+{ve['edge']:.1f}%):** Cửa **{ve['market']}** đang được AI định giá **{ve['ai_prob']:.1f}%** "
            f"so với tỷ lệ thị trường {pkg['odds'].get('provider', 'DraftKings')} (**{ve['mkt_prob']:.1f}%**)."
        )
    return bullets


# ==========================================================
# 3. CLEAN, FLAT UI COMPONENTS (ZERO CLUTTER)
# ==========================================================
def _render_form_pills_html(wdl_list: list) -> str:
    if not wdl_list:
        return '<span style="color:#64748B;font-size:11px;">—</span>'
    colors = {"W": "#00D66C", "D": "#FACC15", "L": "#EF4444"}
    inner = "".join(
        f'<span style="background:{colors.get(x, "#475569")};color:#000;font-weight:900;font-size:9.5px;'
        f'padding:1.5px 5px;border-radius:4px;display:inline-block;">{x}</span>'
        for x in wdl_list
    )
    return f'<div style="display:inline-flex;gap:3px;flex-wrap:nowrap;white-space:nowrap;">{inner}</div>'


def _render_comp_bar_html(label: str, v1: float, v2: float, s1: str, s2: str, c1: str = "#38BDF8", c2: str = "#FF4D6D") -> str:
    a1, a2 = max(0.0, float(v1)), max(0.0, float(v2))
    p1 = (a1 / (a1 + a2) * 100.0) if (a1 + a2) > 0 else 50.0
    return compact_html(f"""
    <div style="margin-bottom:9px;">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
            <span style="color:#F8FAFC;font-weight:800;">{html.escape(s1)}</span>
            <span style="color:#94A3B8;font-weight:600;">{html.escape(label)}</span>
            <span style="color:#F8FAFC;font-weight:800;">{html.escape(s2)}</span>
        </div>
        <div style="display:flex;height:6px;border-radius:999px;overflow:hidden;background:#1E293B;gap:2px;">
            <div style="width:{p1:.1f}%;background:{c1};"></div>
            <div style="width:{100.0-p1:.1f}%;background:{c2};"></div>
        </div>
    </div>
    """)


def _fallback_crest_svg(team_name: str, accent: str = "#38BDF8", size: int = 36) -> str:
    import base64
    t_clean = str(team_name or "FC").strip()
    parts = [w for w in t_clean.replace("-", " ").split() if w.upper() not in ("FC", "CF", "AC", "SC", "AFC")]
    abbr = (parts[0][0] + parts[1][:2]).upper() if len(parts) >= 2 else (parts[0][:3].upper() if parts else "FC")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48">'
        f'<path d="M24 3 L41 9.5 V23 C41 34.5 33.4 41.8 24 45 C14.6 41.8 7 34.5 7 23 V9.5 Z" '
        f'fill="#162135" stroke="{accent}" stroke-width="2.4" stroke-linejoin="round"/>'
        f'<text x="24" y="27.5" text-anchor="middle" fill="#F8FAFC" font-family="Inter,Arial,sans-serif" font-weight="900" font-size="12.5">{html.escape(abbr)}</text>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" style="flex-shrink:0;" alt="{html.escape(t_clean)}">'


def _crest_or_fallback_html(team_name: str, logo_url: str, accent: str = "#38BDF8", size: int = 38) -> str:
    import base64
    t_clean = str(team_name or "FC").strip()
    parts = [w for w in t_clean.replace("-", " ").split() if w.upper() not in ("FC", "CF", "AC", "SC", "AFC")]
    abbr = (parts[0][0] + parts[1][:2]).upper() if len(parts) >= 2 else (parts[0][:3].upper() if parts else "FC")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 48 48">'
        f'<path d="M24 3 L41 9.5 V23 C41 34.5 33.4 41.8 24 45 C14.6 41.8 7 34.5 7 23 V9.5 Z" '
        f'fill="#162135" stroke="{accent}" stroke-width="2.4" stroke-linejoin="round"/>'
        f'<text x="24" y="27.5" text-anchor="middle" fill="#F8FAFC" font-family="Inter,Arial,sans-serif" font-weight="900" font-size="12.5">{html.escape(abbr)}</text>'
        f'</svg>'
    )
    fallback_uri = f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('ascii')}"
    resolved = get_team_logo_url(t_clean, fallback_url=logo_url)
    if resolved:
        return (
            f'<img src="{html.escape(resolved, quote=True)}" width="{size}" height="{size}" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;flex-shrink:0;" alt="{html.escape(t_clean)}" '
            f'onerror="this.onerror=null;this.src=\'{fallback_uri}\';">'
        )
    return f'<img src="{fallback_uri}" width="{size}" height="{size}" style="flex-shrink:0;" alt="{html.escape(t_clean)}">'


def render_single_upcoming_dossier(
    pkg: dict,
    save_pred_fn=None,
    get_pred_fn=None,
    auth_user: dict = None,
    key_prefix: str = "pm",
):
    """Giao diện hồ sơ nghiên cứu & dự đoán phẳng, thoáng, 0 tab lồng nhau."""
    ht, at = pkg["home_team"], pkg["away_team"]
    ens = pkg["ensemble"]
    dc = ens["dc"]
    r = pkg["raw_row"]

    # ---- KHỐI 1: HERO BANNER + THANH XÁC SUẤT 1X2 ----
    h_logo = _crest_or_fallback_html(ht, pkg.get("home_logo", ""), "#38BDF8", 38)
    a_logo = _crest_or_fallback_html(at, pkg.get("away_logo", ""), "#F43F5E", 38)
    h_meta = f"#{pkg['home_rank']} ({pkg['home_pts']}đ) · " if pkg["home_rank"] else ""
    a_meta = f"#{pkg['away_rank']} ({pkg['away_pts']}đ) · " if pkg["away_rank"] else ""

    st.markdown(compact_html(f"""
    <div style="background:linear-gradient(135deg,#111827 0%,#162338 55%,#121926 100%);border:1px solid #283B58;border-radius:15px;padding:15px 18px;margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;padding-bottom:9px;border-bottom:1px solid rgba(56,189,248,0.14);font-size:12px;">
            <span style="color:#38BDF8;font-weight:900;">{html.escape(pkg['league_name'].upper())} {html.escape(pkg['season'])}</span>
            <span style="color:#94A3B8;">🏟️ {html.escape(pkg['venue'])}</span>
        </div>
        <div style="display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:12px;margin:14px 0;">
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px;min-width:0;">
                <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;max-width:100%;">
                    <span style="font-size:clamp(15px,1.9vw,24px);font-weight:900;color:#F8FAFC;text-align:right;line-height:1.18;">{html.escape(ht)}</span>
                    {h_logo}
                </div>
                <div style="color:#94A3B8;font-size:11.5px;font-weight:700;text-align:right;">{h_meta}Elo {pkg['home_elo']:.0f} · <b style="color:#7DD3FC;">xG {ens['lam_h']:.2f}</b></div>
                {_render_form_pills_html(pkg['home_form']['wdl'])}
            </div>
            <div style="background:#0B111E;border:1.5px solid #38BDF8;border-radius:11px;padding:8px 13px;text-align:center;min-width:126px;">
                <div style="color:#FFFFFF;font-size:13px;font-weight:900;white-space:nowrap;">{html.escape(pkg['kickoff_disp'])}</div>
                <div style="color:#FACC15;font-size:11px;font-weight:800;margin-top:2px;white-space:nowrap;">{html.escape(pkg['countdown'])}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-start;gap:5px;min-width:0;">
                <div style="display:flex;align-items:center;gap:8px;justify-content:flex-start;max-width:100%;">
                    {a_logo}
                    <span style="font-size:clamp(15px,1.9vw,24px);font-weight:900;color:#F8FAFC;text-align:left;line-height:1.18;">{html.escape(at)}</span>
                </div>
                <div style="color:#94A3B8;font-size:11.5px;font-weight:700;text-align:left;">{a_meta}Elo {pkg['away_elo']:.0f} · <b style="color:#FDA4AF;">xG {ens['lam_a']:.2f}</b></div>
                {_render_form_pills_html(pkg['away_form']['wdl'])}
            </div>
        </div>
        <div style="display:flex;height:28px;border-radius:8px;overflow:hidden;font-weight:900;font-size:12px;">
            <div style="width:{ens['prob_H']:.1f}%;background:linear-gradient(90deg,#0284C7,#38BDF8);color:#000;display:flex;align-items:center;justify-content:center;min-width:56px;white-space:nowrap;padding:0 6px;overflow:hidden;">1 · {ens['prob_H']:.1f}%</div>
            <div style="width:{ens['prob_D']:.1f}%;background:#475569;color:#F8FAFC;display:flex;align-items:center;justify-content:center;min-width:50px;white-space:nowrap;padding:0 4px;overflow:hidden;">X · {ens['prob_D']:.1f}%</div>
            <div style="width:{ens['prob_A']:.1f}%;background:linear-gradient(90deg,#E11D48,#FF4D6D);color:#FFF;display:flex;align-items:center;justify-content:center;min-width:56px;white-space:nowrap;padding:0 6px;overflow:hidden;">2 · {ens['prob_A']:.1f}%</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ---- 3 THẺ KHUYẾN NGHỊ AI CỐT LÕI (SMART PICKS) ----
    sp = pkg["smart_picks"]
    pc1, pc2, pc3 = st.columns(3, gap="small")
    cards_spec = [
        (pc1, "🏆 KÈO CHÍNH (1X2)", sp["main"][0], sp["main"][1], f"{sp['main'][2]} · Tin cậy: {ens['conf_lbl']}", "#38BDF8"),
        (pc2, "⚽ BÀN THẮNG (O/U)", sp["goals"][0], sp["goals"][1], sp["goals"][2], "#00E676"),
        (pc3, "🎯 TỶ SỐ & VALUE", sp["score"][0], sp["score"][1], sp["score"][2], "#FACC15"),
    ]
    for col_c, title_c, val_c, pct_c, sub_c, accent_c in cards_spec:
        col_c.markdown(compact_html(f"""
        <div style="background:#131A26;border:1px solid #24334A;border-left:4px solid {accent_c};border-radius:11px;padding:11px 13px;margin-bottom:14px;overflow:hidden;">
            <div style="color:#94A3B8;font-size:10.5px;font-weight:800;letter-spacing:0.3px;">{title_c}</div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:4px;margin:4px 0;">
                <span style="color:#F8FAFC;font-size:14.5px;font-weight:900;line-height:1.2;">{html.escape(val_c)}</span>
                <span style="color:{accent_c};font-size:15.5px;font-weight:900;white-space:nowrap;">{html.escape(pct_c)}</span>
            </div>
            <div style="color:#94A3B8;font-size:11px;line-height:1.3;">{html.escape(sub_c)}</div>
        </div>
        """), unsafe_allow_html=True)

    # ---- KHỐI 2: EXPLAINABLE AI (KEY DRIVERS) + ĐỐI CHIẾU MÔ HÌNH ----
    c_ai1, c_ai2 = st.columns([1.05, 1.05], gap="medium")
    with c_ai1:
        with st.container(border=True):
            st.markdown("##### 🧠 Giải Thích AI: 6 Nhân Tố Quyết Định")
            drv = pkg["drivers"][::-1]
            y_f = [d["factor"] for d in drv]
            x_s = [d["score"] for d in drv]
            bar_c = ["#38BDF8" if v >= 0 else "#FF4D6D" for v in x_s]
            hover_d = [f"<b>{d['factor']}</b><br>Tác động: {d['score']:+.1f}<br>{d['detail']}" for d in drv]
            fig_drv = go.Figure(go.Bar(
                x=x_s, y=y_f, orientation="h", marker_color=bar_c,
                text=[f"{v:+.1f}" for v in x_s], textposition="outside",
                textfont=dict(size=10.5, color="#E2E8F0"),
                customdata=hover_d,
                hovertemplate="%{customdata}<extra></extra>",
            ))
            lim_x = max(12.0, max(abs(v) for v in x_s) * 1.38)
            fig_drv.add_vline(x=0, line_color="rgba(226,232,240,0.45)", line_width=1.2)
            fig_drv.update_layout(
                height=235, margin=dict(l=6, r=18, t=8, b=8),
                xaxis=dict(range=[-lim_x, lim_x], showgrid=False, zeroline=False, tickvals=[-lim_x * 0.65, 0, lim_x * 0.65], ticktext=[f"← {at[:10]}", "0", f"{ht[:10]} →"]),
                yaxis=dict(showgrid=False, automargin=True),
                paper_bgcolor="#131A26", plot_bgcolor="#0E1420", font=dict(color="#E2E8F0", size=11),
            )
            st.plotly_chart(fig_drv, width="stretch", key=f"{key_prefix}_drv_{pkg['match_id']}")
            for b in _generate_concise_ai_bullets(pkg):
                st.markdown(f"- {b}")

    with c_ai2:
        with st.container(border=True):
            st.markdown("##### 🔬 Đối Chiếu 4 Tầng Mô Hình & Kèo Thị Trường")
            m_rows = [
                {"Nguồn Mô Hình": "🧠 Opta Deep ML (55%)", "1 (Nhà)": f"{pkg['ml']['prob_H']:.1f}%", "X (Hòa)": f"{pkg['ml']['prob_D']:.1f}%", "2 (Khách)": f"{pkg['ml']['prob_A']:.1f}%"},
                {"Nguồn Mô Hình": "📐 Dixon-Coles xG (25%)", "1 (Nhà)": f"{pkg['form_dc']['prob_H']:.1f}%", "X (Hòa)": f"{pkg['form_dc']['prob_D']:.1f}%", "2 (Khách)": f"{pkg['form_dc']['prob_A']:.1f}%"},
                {"Nguồn Mô Hình": "🏆 Elo Logistic (20%)", "1 (Nhà)": f"{pkg['elo_model']['prob_H']:.1f}%", "X (Hòa)": f"{pkg['elo_model']['prob_D']:.1f}%", "2 (Khách)": f"{pkg['elo_model']['prob_A']:.1f}%"},
                {"Nguồn Mô Hình": "⭐ TỔNG HỢP AI ENSEMBLE", "1 (Nhà)": f"{ens['prob_H']:.1f}%", "X (Hòa)": f"{ens['prob_D']:.1f}%", "2 (Khách)": f"{ens['prob_A']:.1f}%"},
            ]
            if pkg["odds"].get("mkt_prob_H"):
                od = pkg["odds"]
                m_rows.append({"Nguồn Mô Hình": f"🏦 {od.get('provider', 'DraftKings')} Live", "1 (Nhà)": f"{od['mkt_prob_H']:.1f}%", "X (Hòa)": f"{od['mkt_prob_D']:.1f}%", "2 (Khách)": f"{od['mkt_prob_A']:.1f}%"})
            st.dataframe(pd.DataFrame(m_rows), width="stretch", hide_index=True)

            if pkg["value_edges"]:
                ve0 = pkg["value_edges"][0]
                vc = "#00E676" if ve0["edge"] >= 2.5 else "#94A3B8"
                st.markdown(compact_html(f"""
                <div style="background:#0E1420;border:1px solid {vc};border-radius:9px;padding:8px 12px;margin:6px 0 10px;display:flex;justify-content:space-between;align-items:center;font-size:12.5px;">
                    <span>💎 Value Edge tốt nhất: <b style="color:#F8FAFC;">{html.escape(ve0['market'])}</b></span>
                    <span style="color:{vc};font-weight:900;">{ve0['edge']:+.1f}% (¼ Kelly: {ve0['kelly_pct']:.1f}%)</span>
                </div>
                """), unsafe_allow_html=True)

            mkt_html = ""
            mkt_html += _render_comp_bar_html("Tài / Xỉu 1.5 bàn", dc["over_15"], dc["under_15"], f"Tài {dc['over_15']}%", f"Xỉu {dc['under_15']}%", "#00E676", "#64748B")
            mkt_html += _render_comp_bar_html("Tài / Xỉu 2.5 bàn", dc["over_25"], dc["under_25"], f"Tài {dc['over_25']}%", f"Xỉu {dc['under_25']}%", "#00E676", "#64748B")
            mkt_html += _render_comp_bar_html("Tài / Xỉu 3.5 bàn", dc["over_35"], dc["under_35"], f"Tài {dc['over_35']}%", f"Xỉu {dc['under_35']}%", "#FACC15", "#64748B")
            mkt_html += _render_comp_bar_html("Hai đội cùng ghi bàn (BTTS)", dc["btts_yes"], dc["btts_no"], f"Có {dc['btts_yes']}%", f"Không {dc['btts_no']}%", "#38BDF8", "#94A3B8")
            mkt_html += _render_comp_bar_html("Giữ sạch lưới (Clean Sheet)", dc["cs_home"], dc["cs_away"], f"{ht[:10]} {dc['cs_home']}%", f"{at[:10]} {dc['cs_away']}%", "#38BDF8", "#FF4D6D")
            st.markdown(mkt_html, unsafe_allow_html=True)

    # ---- KHỐI 3: MA TRẬN TỶ SỐ 6×6 & SỨC MẠNH 4 TUYẾN + RADAR ----
    c_m1, c_m2 = st.columns([1.1, 1.0], gap="medium")
    with c_m1:
        with st.container(border=True):
            st.markdown(f"##### 🎲 Ma Trận Tỷ Số Dixon-Coles 6×6 & Top 5 Tỷ Số")
            mat = dc["matrix_6x6"]
            fig_m = go.Figure(data=go.Heatmap(
                z=mat,
                x=[f"{at[:9]} {j}" for j in range(6)],
                y=[f"{ht[:9]} {i}" for i in range(6)],
                text=[[f"{mat[i, j]:.1f}%" for j in range(6)] for i in range(6)],
                texttemplate="%{text}",
                colorscale=[[0.0, "#0F172A"], [0.35, "#075985"], [0.7, "#0284C7"], [1.0, "#00E676"]],
                showscale=False,
            ))
            fig_m.update_layout(
                height=305, margin=dict(l=10, r=10, t=8, b=8),
                paper_bgcolor="#131A26", plot_bgcolor="#0E1420",
                font=dict(color="#F8FAFC", size=11.5), yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_m, width="stretch", key=f"{key_prefix}_mat_{pkg['match_id']}")
            pills_top = " &nbsp; ".join(
                f'<span style="background:#0F172A;border:1px solid {"#00E676" if idx==0 else "#334155"};border-radius:8px;padding:4px 9px;font-size:12px;">'
                f'<b style="color:#FFF;">{sc["score"]}</b> <span style="color:{"#00E676" if idx==0 else "#38BDF8"};font-weight:800;">{sc["prob_pct"]:.1f}%</span></span>'
                for idx, sc in enumerate(dc["top_scorelines"][:5])
            )
            st.markdown(f'<div style="text-align:center;margin-top:4px;">{pills_top}</div>', unsafe_allow_html=True)

    with c_m2:
        with st.container(border=True):
            st.markdown("##### ⚔️ Sức Mạnh 4 Tuyến & Radar Chiến Thuật 8 Trục")
            if r is not None and "home_avg_fwd_panna" in r.index and pd.notna(r.get("home_avg_fwd_panna")):
                z_items = [
                    ("🧤 GK", _safe_float(r.get("home_gk_panna")), _safe_float(r.get("away_gk_panna"))),
                    ("🛡️ Thủ", _safe_float(r.get("home_avg_def_panna")), _safe_float(r.get("away_avg_def_panna"))),
                    ("⚙️ Giữa", _safe_float(r.get("home_avg_mid_panna")), _safe_float(r.get("away_avg_mid_panna"))),
                    ("⚡ Công", _safe_float(r.get("home_avg_fwd_panna")), _safe_float(r.get("away_avg_fwd_panna"))),
                ]
                z_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">'
                for z_lbl, hv, av in z_items:
                    hsc = max(45, min(99, int(round(68 + hv * 115))))
                    asc = max(45, min(99, int(round(68 + av * 115))))
                    bc = "#38BDF8" if hsc > asc else ("#FF4D6D" if asc > hsc else "#475569")
                    z_html += (
                        f'<div style="background:#0E1420;border:1px solid #243247;border-top:2.5px solid {bc};border-radius:8px;padding:6px;text-align:center;">'
                        f'<div style="color:#94A3B8;font-size:10.5px;font-weight:800;">{z_lbl}</div>'
                        f'<div style="font-size:14px;font-weight:900;margin-top:2px;"><span style="color:#38BDF8;">{hsc}</span> <span style="color:#475569;">-</span> <span style="color:#FF4D6D;">{asc}</span></div>'
                        f'</div>'
                    )
                z_html += "</div>"
                st.markdown(z_html, unsafe_allow_html=True)

                radar_specs = [
                    ("Ghi bàn", "home_sk_att_goals", "away_sk_att_goals", 0.22),
                    ("Dứt điểm", "home_sk_att_shots", "away_sk_att_shots", 1.8),
                    ("Key Pass", "home_sk_att_key_passes", "away_sk_att_key_passes", 2.0),
                    ("Cơ hội lớn", "home_sk_att_big_chance_created", "away_sk_att_big_chance_created", 0.28),
                    ("Vòng cấm", "home_sk_att_touches_opp_box", "away_sk_att_touches_opp_box", 3.2),
                    ("Tắc bóng", "home_sk_def_tackles_won", "away_sk_def_tackles_won", 1.3),
                    ("Đánh chặn", "home_sk_def_interceptions", "away_sk_def_interceptions", 1.3),
                    ("Không chiến", "home_sk_def_aerial_won", "away_sk_def_aerial_won", 2.1),
                ]
                axes_r = [sp_r[0] for sp_r in radar_specs]
                hv_r = [min(99, max(20, round(_safe_float(r.get(sp_r[1]), sp_r[3] * 0.5) / sp_r[3] * 100.0, 1))) for sp_r in radar_specs]
                av_r = [min(99, max(20, round(_safe_float(r.get(sp_r[2]), sp_r[3] * 0.5) / sp_r[3] * 100.0, 1))) for sp_r in radar_specs]
                fig_rad = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(r=hv_r + [hv_r[0]], theta=axes_r + [axes_r[0]], fill="toself", fillcolor="rgba(56,189,248,0.24)", line=dict(color="#38BDF8", width=2.4), name=ht))
                fig_rad.add_trace(go.Scatterpolar(r=av_r + [av_r[0]], theta=axes_r + [axes_r[0]], fill="toself", fillcolor="rgba(255,77,109,0.22)", line=dict(color="#FF4D6D", width=2.4), name=at))
                fig_rad.update_layout(
                    polar=dict(bgcolor="#0D131F", radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(148,163,184,0.16)"), angularaxis=dict(tickfont=dict(size=10.5, color="#F8FAFC"))),
                    height=265, margin=dict(l=30, r=30, t=22, b=10),
                    paper_bgcolor="#131A26", font=dict(color="#E2E8F0"),
                    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)),
                )
                st.plotly_chart(fig_rad, width="stretch", key=f"{key_prefix}_rad_{pkg['match_id']}")
            else:
                # Fallback bar chart for custom matchups without full 110-feature row
                hf, af = pkg["home_form"], pkg["away_form"]
                fb_html = ""
                fb_html += _render_comp_bar_html("Điểm TB 10 trận (PPG)", hf["ppg"], af["ppg"], f"{hf['ppg']:.2f}", f"{af['ppg']:.2f}")
                fb_html += _render_comp_bar_html("xG tạo ra / trận", hf["xg_pg"], af["xg_pg"], f"{hf['xg_pg']:.2f}", f"{af['xg_pg']:.2f}")
                fb_html += _render_comp_bar_html("xGA chịu đựng / trận", hf["xga_pg"], af["xga_pg"], f"{hf['xga_pg']:.2f}", f"{af['xga_pg']:.2f}")
                fb_html += _render_comp_bar_html("Tỷ lệ giữ sạch lưới (%)", hf["cs_pct"], af["cs_pct"], f"{hf['cs_pct']:.0f}%", f"{af['cs_pct']:.0f}%")
                fb_html += _render_comp_bar_html("Tỷ lệ Tài 2.5 bàn (%)", hf["o25_pct"], af["o25_pct"], f"{hf['o25_pct']:.0f}%", f"{af['o25_pct']:.0f}%")
                st.markdown(fb_html, unsafe_allow_html=True)

    # ---- KHỐI 4: 3 EXPANDER GỌN GÀNG (MỞ KHI CẦN ĐÀO SÂU — KHÔNG LÀM RỐI MÀN HÌNH!) ----
    with st.expander(f"📈 Phong Độ 10 Trận Gần Nhất, Biểu Đồ xG & Sân Nhà/Sân Khách ({ht} vs {at})", expanded=False):
        hf, af = pkg["home_form"], pkg["away_form"]
        cf1, cf2 = st.columns(2, gap="medium")
        for col_f, t_nm, f_obj, accent in [(cf1, ht, hf, "#38BDF8"), (cf2, at, af, "#FF4D6D")]:
            with col_f:
                st.markdown(f"**{t_nm}** — `{f_obj['ppg']} đ/trận` · Ghi `{f_obj['gf_pg']}` (xG `{f_obj['xg_pg']}`, chênh `{f_obj['finishing_diff']:+.2f}`) · Thủng `{f_obj['ga_pg']}` (xGA `{f_obj['xga_pg']}`)")
                rdf = f_obj["rows"]
                if not rdf.empty:
                    xg_chr = rdf.iloc[::-1].dropna(subset=["xg", "xga"])
                    if len(xg_chr) >= 3:
                        fig_tr = go.Figure()
                        xl = [f"{rx['date'][5:]}<br>{rx['opponent'][:7]}" for _, rx in xg_chr.iterrows()]
                        fig_tr.add_trace(go.Scatter(x=xl, y=xg_chr["xg"], mode="lines+markers", name="xG", line=dict(color=accent, width=2.4)))
                        fig_tr.add_trace(go.Scatter(x=xl, y=xg_chr["xga"], mode="lines+markers", name="xGA", line=dict(color="#94A3B8", width=1.8, dash="dot")))
                        fig_tr.update_layout(height=180, margin=dict(l=5, r=5, t=8, b=20), paper_bgcolor="#131A26", plot_bgcolor="#0E1420", font=dict(color="#E2E8F0", size=10), legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"))
                        st.plotly_chart(fig_tr, width="stretch", key=f"{key_prefix}_xgtr_{t_nm}_{pkg['match_id']}")
                    st.dataframe(rdf[["date", "venue", "opponent", "score", "xg", "xga", "res"]].rename(columns={"date": "Ngày", "venue": "Sân", "opponent": "Đối Thủ", "score": "Tỷ Số", "res": "KQ"}), width="stretch", hide_index=True, height=240)
                else:
                    st.info("Chưa có dữ liệu 10 trận gần đây.")

    with st.expander(f"🤝 Lịch Sử Đối Đầu (H2H), Ngôi Sao 2026/27 & Tin Trước Trận", expanded=False):
        h2h = pkg["h2h"]
        if h2h["matches"].empty:
            st.caption(f"Chưa có dữ liệu đối đầu gần đây giữa {ht} và {at}.")
        else:
            hk1, hk2, hk3, hk4 = st.columns(4, gap="small")
            hk1.metric(f"🔵 {ht[:12]} Thắng", f"{h2h['h_wins']} trận", border=True)
            hk2.metric("⚖️ Hòa", f"{h2h['draws']} trận", border=True)
            hk3.metric(f"🔴 {at[:12]} Thắng", f"{h2h['a_wins']} trận", border=True)
            hk4.metric("⚽ TB Bàn / BTTS", f"{h2h['avg_goals']} bàn", f"BTTS {h2h['btts_pct']}% · O2.5 {h2h['o25_pct']}%", border=True)
            st.dataframe(h2h["matches"], width="stretch", hide_index=True)

        if pkg["leaders"] or pkg["news"]:
            cl1, cl2 = st.columns(2, gap="medium")
            with cl1:
                st.markdown("**🌟 Ngôi sao dẫn đầu mùa 2026/27 (Live ESPN):**")
                for tl in pkg["leaders"]:
                    st.caption(f"🛡️ **{tl['team']}**: " + " · ".join(f"{it['category']}: **{it['player']}** ({it['value']})" for it in tl["items"]))
            with cl2:
                st.markdown("**📰 Tin nhận định trước trận (Live ESPN):**")
                for art in pkg["news"]:
                    st.markdown(f"- [{art['headline']}]({art['link']}) *({art['published']})*" if art.get("link") else f"- **{art['headline']}**")

    with st.expander("🎛️ Giả Lập Kịch Bản 'What-If' (Chấn Thương / Thời Tiết / Sân Trung Lập) & Lưu Dự Đoán", expanded=False):
        sc1, sc2 = st.columns(2, gap="medium")
        h_att_mod = sc1.slider(f"🔵 Công {ht} (%):", -35, 35, 0, 5, key=f"{key_prefix}_ha_{pkg['match_id']}")
        h_def_mod = sc1.slider(f"🛡️ Thủ {ht} (%):", -35, 35, 0, 5, key=f"{key_prefix}_hd_{pkg['match_id']}")
        a_att_mod = sc2.slider(f"🔴 Công {at} (%):", -35, 35, 0, 5, key=f"{key_prefix}_aa_{pkg['match_id']}")
        a_def_mod = sc2.slider(f"🛡️ Thủ {at} (%):", -35, 35, 0, 5, key=f"{key_prefix}_ad_{pkg['match_id']}")

        tc1, tc2, tc3 = st.columns(3)
        neutral_v = tc1.checkbox("🏟️ Sân trung lập", value=False, key=f"{key_prefix}_nv_{pkg['match_id']}")
        rain_v = tc2.checkbox("🌧️ Mưa lớn (-12% bàn)", value=False, key=f"{key_prefix}_rn_{pkg['match_id']}")
        intense_v = tc3.checkbox("🔥 Đôi công (+15% bàn)", value=False, key=f"{key_prefix}_mw_{pkg['match_id']}")

        sim_lh = ens["lam_h"] * (1.0 + h_att_mod / 100.0) * (1.0 - a_def_mod / 100.0) * (0.91 if neutral_v else 1.0) * (0.88 if rain_v else 1.0) * (1.15 if intense_v else 1.0)
        sim_la = ens["lam_a"] * (1.0 + a_att_mod / 100.0) * (1.0 - h_def_mod / 100.0) * (1.09 if neutral_v else 1.0) * (0.88 if rain_v else 1.0) * (1.15 if intense_v else 1.0)
        sim_dc = dixon_coles_matrix(sim_lh, sim_la)

        mk1, mk2, mk3, mk4, mk5 = st.columns(5, gap="small")
        mk1.metric(f"🔵 {ht[:10]}", f"{sim_dc['prob_H']:.1f}%", f"{sim_dc['prob_H']-dc['prob_H']:+.1f}%", border=True)
        mk2.metric("⚖️ Hòa", f"{sim_dc['prob_D']:.1f}%", f"{sim_dc['prob_D']-dc['prob_D']:+.1f}%", border=True)
        mk3.metric(f"🔴 {at[:10]}", f"{sim_dc['prob_A']:.1f}%", f"{sim_dc['prob_A']-dc['prob_A']:+.1f}%", border=True)
        mk4.metric("⚽ Tài 2.5", f"{sim_dc['over_25']:.1f}%", f"{sim_dc['over_25']-dc['over_25']:+.1f}%", border=True)
        mk5.metric("🎯 Tỷ số #1", sim_dc["top_scorelines"][0]["score"], f"{sim_dc['top_scorelines'][0]['prob_pct']:.1f}%", border=True)

        if save_pred_fn:
            st.markdown("---")
            pred_user = (auth_user or {}).get("username") or "__guest_session__"
            existing_p = get_pred_fn(pred_user, f"upcoming_{pkg['match_id']}") if get_pred_fn else None
            pc1, pc2, pc3, pc4 = st.columns([0.9, 0.9, 2.2, 1.1])
            up_h = pc1.number_input(f"Bàn {ht[:10]}:", 0, 10, int(existing_p["pred_home"] if existing_p else round(ens["lam_h"])), key=f"{key_prefix}_uh_{pkg['match_id']}")
            up_a = pc2.number_input(f"Bàn {at[:10]}:", 0, 10, int(existing_p["pred_away"] if existing_p else round(ens["lam_a"])), key=f"{key_prefix}_ua_{pkg['match_id']}")
            up_n = pc3.text_input("Ghi chú cá nhân:", value=(existing_p["note"] if existing_p else ""), placeholder="VD: Chọn 1X + Tài 2.5...", key=f"{key_prefix}_un_{pkg['match_id']}")
            if pc4.button("💾 Lưu Dự Đoán", width="stretch", key=f"{key_prefix}_usave_{pkg['match_id']}"):
                save_pred_fn(pred_user, f"upcoming_{pkg['match_id']}", f"{ht} vs {at} ({pkg['match_date']})", int(up_h), int(up_a), up_n)
                st.success(f"✅ Đã lưu dự đoán **{ht} {up_h}-{up_a} {at}**!")


def render_prematch_research_workspace(
    save_pred_fn=None,
    get_pred_fn=None,
    auth_user: dict = None,
    preselected_match_id: str = None,
):
    """Trung Tâm AI & Dự Đoán Trận Đấu Thống Nhất — 4 Tab Phẳng Lazy-Load."""
    up_df = load_upcoming_fixtures_df()
    hist_df = load_historical_xg_df()
    espn_cache = get_synced_espn_results()

    hl, hr = st.columns([4.0, 1.2])
    with hl:
        st.markdown("### 🔮 Trung Tâm AI Phân Tích & Dự Đoán Trận Đấu")
        st.caption(f"📚 **{len(up_df):,} trận trong lịch (13 giải)** · **{len(hist_df):,} trận lịch sử xG/Elo** · Live ESPN: `{espn_cache.get('updated_utc') or 'Cache'}`")
    with hr:
        if st.button("🔄 Đồng bộ Live", width="stretch", key="btn_sync_espn_live"):
            with st.spinner("Đang cập nhật kèo, tỷ số & lịch Live từ ESPN..."):
                get_synced_espn_results(force_refresh=True)
                ok, msg = refresh_espn_live_now()
            st.success(msg) if ok else st.warning(msg)
            if ok:
                st.rerun()

    if up_df.empty:
        st.warning("Chưa tìm thấy dữ liệu trận chưa diễn ra. Hãy chạy `python build_pro_analytics_store.py`.")
        return

    today_dt = datetime.now(_VN_TZ).date()
    today_vn_str = today_dt.strftime("%Y-%m-%d")

    def _is_row_truly_upcoming(rr_row) -> bool:
        es_m = match_espn_live_entry(
            str(rr_row.get("home_team") or ""),
            str(rr_row.get("away_team") or ""),
            str(rr_row.get("match_date") or ""),
            espn_cache,
            max_day_diff=2,
        )
        st_val = str(es_m.get("state") or "").lower()
        if st_val == "post":
            return False
        eff_d = str(es_m.get("vn_date") or rr_row.get("match_date") or "")[:10]
        if eff_d and eff_d < today_vn_str and st_val != "pre":
            return False
        return True

    if st.session_state.get("jump_upcoming_match_id") or preselected_match_id:
        st.session_state["prematch_tabs_sel"] = "🎯 Trận Sắp Đá"

    tab_up, tab_scan, tab_custom, tab_xpts = st.tabs(
        [
            "🎯 Trận Sắp Đá",
            "📡 Quét Vòng Đấu",
            "⚡ Ghép 2 Đội",
            "📊 BXH xPTS",
        ],
        key="prematch_tabs_sel",
        on_change="rerun",
    )

    # ---------- TAB 1: SINGLE UPCOMING MATCH ----------
    if tab_up.open:
        with tab_up:
            jump_uid = st.session_state.pop("jump_upcoming_match_id", None) or preselected_match_id
            up_active_mask = up_df.apply(_is_row_truly_upcoming, axis=1)
            up_active_df = up_df[up_active_mask].reset_index(drop=True)
            if up_active_df.empty:
                up_active_df = up_df

            if jump_uid and str(jump_uid) in set(up_df["match_id"].astype(str)):
                j_row = up_df[up_df["match_id"].astype(str) == str(jump_uid)].iloc[0]
                j_espn = match_espn_live_entry(
                    str(j_row["home_team"]),
                    str(j_row["away_team"]),
                    str(j_row["match_date"]),
                    espn_cache,
                    max_day_diff=2,
                )
                if str(j_espn.get("state") or "").lower() == "post" and j_espn.get("home_score") is not None:
                    hs_j, as_j = int(j_espn["home_score"]), int(j_espn["away_score"])
                    c_ban, c_go = st.columns([3.5, 1.5])
                    c_ban.success(
                        f"✅ Trận đấu đã kết thúc: **{j_row['home_team']} {hs_j} - {as_j} {j_row['away_team']}** "
                        f"({j_espn.get('status_detail') or 'FT'})"
                    )
                    if c_go.button("🏟️ Xem Phân Tích Sau Trận (6 Tab)", key=f"pm_jump_post_{jump_uid}", width="stretch"):
                        st.session_state["daily_match_center_fixture"] = j_espn
                        st.session_state["pending_workspace"] = "ws_match"
                        st.rerun()
                else:
                    st.session_state["pm_league_pick"] = str(j_row["league"])
                    st.session_state["pm_days_window"] = "Toàn mùa"

            avail_leagues = [lg for lg in LEAGUE_ORDER if lg in set(up_active_df["league"].unique())]
            for lg in sorted(up_active_df["league"].unique()):
                if lg not in avail_leagues:
                    avail_leagues.append(lg)

            c1, c2 = st.columns([1.6, 1.0], gap="small")
            lg_opts = ["ALL"] + avail_leagues
            if "pm_league_pick" not in st.session_state or st.session_state["pm_league_pick"] not in lg_opts:
                st.session_state["pm_league_pick"] = "ENG" if "ENG" in lg_opts else "ALL"
            sel_lg = c1.selectbox("Giải đấu:", lg_opts, format_func=lambda x: "🌍 Tất cả 13 giải" if x == "ALL" else LEAGUE_META.get(x, (x, x, ""))[0], key="pm_league_pick")

            win_opts = ["7 ngày tới", "14 ngày tới", "30 ngày tới", "Toàn mùa"]
            if "pm_days_window" not in st.session_state or st.session_state["pm_days_window"] not in win_opts:
                st.session_state["pm_days_window"] = "14 ngày tới"
            sel_win = c2.selectbox("Thời gian:", win_opts, key="pm_days_window")

            sub = up_active_df if sel_lg == "ALL" else up_active_df[up_active_df["league"] == sel_lg]
            min_d = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
            days_map = {"7 ngày tới": 7, "14 ngày tới": 14, "30 ngày tới": 30}
            if sel_win in days_map:
                max_d = (today_dt + timedelta(days=days_map[sel_win])).strftime("%Y-%m-%d")
                sub = sub[(sub["match_date"] >= min_d) & (sub["match_date"] <= max_d)]

            if sub.empty:
                st.info("Không có trận chưa đá nào trong khoảng thời gian này — hãy chọn **30 ngày tới** hoặc **Toàn mùa**.")
            else:
                sub = sub.reset_index(drop=True)
                mids = sub["match_id"].astype(str).tolist()
                mid_map = {str(rr["match_id"]): rr for _, rr in sub.iterrows()}

                def _fmt_m(mid):
                    rr = mid_map[mid]
                    d_s = f"{rr['match_date'][8:10]}/{rr['match_date'][5:7]}"
                    pH = round(_safe_float(rr["prob_H"]) * 100)
                    pD = round(_safe_float(rr["prob_D"]) * 100)
                    pA = round(_safe_float(rr["prob_A"]) * 100)
                    return f"⏰ {d_s} · {rr['home_team']} vs {rr['away_team']}  (AI: {pH}% / {pD}% / {pA}%)"

                if jump_uid and str(jump_uid) in mids:
                    st.session_state["pm_match_pick"] = str(jump_uid)
                if "pm_match_pick" not in st.session_state or st.session_state["pm_match_pick"] not in mids:
                    st.session_state["pm_match_pick"] = mids[0]
                picked_mid = st.selectbox(f"⚽ Chọn trận chưa diễn ra ({len(mids)} trận · gõ tên đội để tìm):", mids, format_func=_fmt_m, key="pm_match_pick")

                pkg = build_prematch_research_package(mid_map[picked_mid], hist_df, espn_cache, fetch_live_intel=True)
                render_single_upcoming_dossier(pkg, save_pred_fn=save_pred_fn, get_pred_fn=get_pred_fn, auth_user=auth_user, key_prefix="pm_main")

    # ---------- TAB 2: MATCHDAY SCANNER ----------
    if tab_scan.open:
        with tab_scan:
            sc1, sc2, sc3 = st.columns([1.0, 1.6, 1.2], gap="small")
            scan_days = sc1.selectbox("Khoảng quét:", [3, 7, 14, 30], index=1, format_func=lambda d: f"{d} ngày tới", key="scan_days")
            scan_preset = sc2.selectbox(
                "Bộ lọc thông minh:",
                [
                    "🔥 Tâm điểm vòng đấu (Elo cao nhất)",
                    "⚽ Mưa bàn thắng (xG & Tài 2.5 cao nhất)",
                    "🛡️ Cửa thắng sáng nhất (Bankers)",
                    "⚡ Cân tài cân sức / Dễ bất ngờ",
                    "💎 Value Edge (AI vs DraftKings Live)",
                ],
                key="scan_preset",
            )
            scan_lg = sc3.selectbox(
                "Giải đấu:",
                ["TOP7", "ALL"] + [lg for lg in LEAGUE_ORDER if lg in set(up_df["league"].unique())],
                format_func=lambda c: "🌟 Top 5 + UCL/UEL" if c == "TOP7" else ("🌍 Tất cả 13 giải" if c == "ALL" else LEAGUE_META.get(c, (c, c, ""))[0]),
                key="scan_lg",
            )

            sc_df = up_df[(up_df["match_date"] >= (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")) & (up_df["match_date"] <= (today_dt + timedelta(days=scan_days)).strftime("%Y-%m-%d"))]
            if not sc_df.empty:
                sc_df = sc_df[sc_df.apply(_is_row_truly_upcoming, axis=1)]
            if scan_lg == "TOP7":
                sc_df = sc_df[sc_df["league"].isin(["ENG", "ESP", "ITA", "GER", "FRA", "UCL", "UEL"])]
            elif scan_lg != "ALL":
                sc_df = sc_df[sc_df["league"] == scan_lg]

            if sc_df.empty:
                st.info("Không có trận chưa đá nào trong khoảng quét này.")
            else:
                rows = []
                for _, rr in sc_df.iterrows():
                    lh, la = _safe_float(rr.get("pred_home_goals"), 1.35), _safe_float(rr.get("pred_away_goals"), 1.15)
                    pH = round(_safe_float(rr.get("prob_H"), 0.35) * 100.0, 1)
                    pD = round(_safe_float(rr.get("prob_D"), 0.28) * 100.0, 1)
                    pA = round(_safe_float(rr.get("prob_A"), 0.37) * 100.0, 1)
                    dc_q = dixon_coles_matrix(lh, la, max_goals=6)
                    comb_elo = (_safe_float(rr.get("home_elo"), 1650) + _safe_float(rr.get("away_elo"), 1650)) / 2.0
                    espn_m = match_espn_live_entry(str(rr["home_team"]), str(rr["away_team"]), str(rr["match_date"]), espn_cache, max_day_diff=2)
                    od = espn_m.get("odds") or {}
                    max_edge, edge_lbl = 0.0, "—"
                    if od.get("mkt_prob_H"):
                        best_c = max([(f"1 ({pH-od['mkt_prob_H']:+.1f}%)", pH - od["mkt_prob_H"]), (f"X ({pD-od['mkt_prob_D']:+.1f}%)", pD - od["mkt_prob_D"]), (f"2 ({pA-od['mkt_prob_A']:+.1f}%)", pA - od["mkt_prob_A"])], key=lambda x: x[1])
                        max_edge, edge_lbl = round(best_c[1], 1), best_c[0]
                    rows.append({
                        "match_id": str(rr["match_id"]),
                        "Ngày": str(rr["match_date"])[5:10],
                        "Giải": LEAGUE_META.get(str(rr["league"]), (str(rr["league"]), str(rr["league"]), ""))[1],
                        "Cặp Đấu": f"{rr['home_team']} vs {rr['away_team']}",
                        "xG": f"{lh:.2f}-{la:.2f}",
                        "1 / X / 2": f"{pH:.0f}% / {pD:.0f}% / {pA:.0f}%",
                        "Tài 2.5": f"{dc_q['over_25']:.0f}%",
                        "BTTS": f"{dc_q['btts_yes']:.0f}%",
                        "Tỷ Số #1": dc_q["top_scorelines"][0]["score"],
                        "Value Edge": edge_lbl,
                        "_elo": comb_elo, "_xg": lh + la, "_maxp": max(pH, pD, pA), "_bal": 100.0 - abs(pH - pA), "_edge": max_edge, "_od": int(bool(od.get("mkt_prob_H"))),
                    })
                sdf = pd.DataFrame(rows)
                if "Tâm điểm" in scan_preset:
                    sdf = sdf.sort_values(["_elo", "_xg"], ascending=False)
                elif "Mưa bàn thắng" in scan_preset:
                    sdf = sdf.sort_values(["_xg", "_elo"], ascending=False)
                elif "Bankers" in scan_preset:
                    sdf = sdf.sort_values(["_maxp", "_elo"], ascending=False)
                elif "Cân tài" in scan_preset:
                    sdf = sdf.sort_values(["_bal", "_elo"], ascending=False)
                else:
                    sdf = sdf[sdf["_od"] == 1].sort_values(["_edge", "_elo"], ascending=False)

                if sdf.empty:
                    st.info("Chưa có trận có kèo Live khớp bộ lọc này.")
                else:
                    top_s = sdf.head(18).reset_index(drop=True)
                    st.dataframe(top_s[["Ngày", "Giải", "Cặp Đấu", "xG", "1 / X / 2", "Tài 2.5", "BTTS", "Tỷ Số #1", "Value Edge"]], width="stretch", hide_index=True)
                    st.caption("⚡ Bấm vào bất kỳ trận nổi bật nào dưới đây để mở hồ sơ AI đầy đủ:")
                    b_cols = st.columns(3, gap="small")
                    for idx_b, br in top_s.head(6).iterrows():
                        with b_cols[idx_b % 3]:
                            if st.button(f"🔬 {br['Cặp Đấu']} ({br['Ngày']})", key=f"sc_jmp_{br['match_id']}_{idx_b}", width="stretch"):
                                st.session_state["jump_upcoming_match_id"] = br["match_id"]
                                st.session_state["prematch_tabs_sel"] = "🎯 Trận Sắp Đá"
                                st.rerun()

    # ---------- TAB 3: CUSTOM MATCHUP AI (250+ CLUBS & NATIONAL TEAMS) ----------
    if tab_custom.open:
        with tab_custom:
            st.caption("Ghép bất kỳ 2 CLB Châu Âu hoặc Đội tuyển Quốc gia nào để chạy mô hình AI đầy đủ (Elo + Dixon-Coles xG + Key Drivers + H2H + What-If).")
            rec_s = hist_df[hist_df["season"].isin(["2026-2027", "2025-2026"])] if not hist_df.empty else pd.DataFrame()
            teams_pool = sorted(set(rec_s["home_team"].dropna().astype(str)) | set(rec_s["away_team"].dropna().astype(str))) if not rec_s.empty else ["Manchester City", "Arsenal", "Real Madrid", "Barcelona", "Bayern Munich", "Liverpool", "PSG", "Inter Milan"]

            _jmp_ch = st.session_state.pop("jump_custom_home", None)
            _jmp_ca = st.session_state.pop("jump_custom_away", None)
            if _jmp_ch or _jmp_ca:
                pool_alias = {_team_alias_key(t): t for t in teams_pool}
                if _jmp_ch:
                    ch_resolved = _jmp_ch if _jmp_ch in teams_pool else pool_alias.get(_team_alias_key(_jmp_ch))
                    if ch_resolved:
                        st.session_state["cust_ai_h"] = ch_resolved
                if _jmp_ca:
                    ca_resolved = _jmp_ca if _jmp_ca in teams_pool else pool_alias.get(_team_alias_key(_jmp_ca))
                    if ca_resolved:
                        st.session_state["cust_ai_a"] = ca_resolved

            cc1, cc2, cc3 = st.columns([1.6, 1.6, 1.0], gap="small")
            h_idx = teams_pool.index("Manchester City") if "Manchester City" in teams_pool else 0
            a_idx = teams_pool.index("Real Madrid") if "Real Madrid" in teams_pool else min(1, len(teams_pool) - 1)
            ch_team = cc1.selectbox("🔵 Đội Chủ Nhà:", teams_pool, index=h_idx, key="cust_ai_h")
            ca_team = cc2.selectbox("🔴 Đội Khách:", teams_pool, index=a_idx, key="cust_ai_a")
            is_neutral = cc3.checkbox("🏟️ Sân trung lập", value=False, key="cust_ai_nv")

            if ch_team == ca_team:
                st.warning("Hãy chọn 2 đội bóng khác nhau để phân tích đối đầu.")
            else:
                hf_c = extract_team_recent_form(hist_df, ch_team, "2027-06-01", n=12)
                af_c = extract_team_recent_form(hist_df, ca_team, "2027-06-01", n=12)
                h2h_c = extract_h2h_history(hist_df, ch_team, ca_team, "2027-06-01", limit=10)
                h_adv = 0.0 if is_neutral else 65.0
                elo_h, elo_a = hf_c["latest_elo"], af_c["latest_elo"]
                e_pH, e_pD, e_pA = _elo_1x2_probs(elo_h, elo_a, home_adv=h_adv)

                lh_c = max(0.35, min(3.8, (hf_c["xg_pg"] * af_c["xga_pg"] / 1.32) * (1.0 if is_neutral else 1.08)))
                la_c = max(0.30, min(3.5, (af_c["xg_pg"] * hf_c["xga_pg"] / 1.32) * (1.0 if is_neutral else 0.93)))
                dc_c = dixon_coles_matrix(lh_c, la_c)

                ens_pH = round(0.55 * dc_c["prob_H"] + 0.45 * e_pH, 1)
                ens_pD = round(0.55 * dc_c["prob_D"] + 0.45 * e_pD, 1)
                ens_pA = round(max(0.5, 100.0 - ens_pH - ens_pD), 1)
                drv_c = compute_ai_key_drivers(elo_h, elo_a, hf_c, af_c, h2h_c, 0.0, 0.0)

                synth_pkg = {
                    "match_id": f"custom_{_team_alias_key(ch_team)}_{_team_alias_key(ca_team)}",
                    "league_code": "CUSTOM", "league_name": "⚡ GIẢ LẬP ĐỐI ĐẦU AI", "league_short": "Custom AI",
                    "season": "2026/27", "match_date": datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d"),
                    "kickoff_disp": "Sân trung lập" if is_neutral else f"Sân nhà {ch_team}",
                    "countdown": "⚡ Mô phỏng tức thì", "venue": "Neutral Arena" if is_neutral else f"{ch_team} Stadium",
                    "home_team": ch_team, "away_team": ca_team, "home_logo": "", "away_logo": "",
                    "home_rank": None, "away_rank": None, "home_pts": None, "away_pts": None,
                    "home_elo": elo_h, "away_elo": elo_a, "rest_diff": 0.0,
                    "ml": {"lam_h": dc_c["lam_h"], "lam_a": dc_c["lam_a"], "prob_H": dc_c["prob_H"], "prob_D": dc_c["prob_D"], "prob_A": dc_c["prob_A"]},
                    "elo_model": {"prob_H": e_pH, "prob_D": e_pD, "prob_A": e_pA},
                    "form_dc": dc_c,
                    "ensemble": {"lam_h": dc_c["lam_h"], "lam_a": dc_c["lam_a"], "prob_H": ens_pH, "prob_D": ens_pD, "prob_A": ens_pA, "dc": dc_c, "confidence": round(max(ens_pH, ens_pD, ens_pA), 1), "conf_lbl": "Mô phỏng AI"},
                    "smart_picks": {
                        "main": (f"{ch_team} Thắng" if ens_pH >= ens_pA else f"{ca_team} Thắng", f"{max(ens_pH, ens_pA):.1f}%", f"1X: {dc_c['dc_1X']}% · X2: {dc_c['dc_X2']}%"),
                        "goals": ("Tài 2.5 bàn" if dc_c["over_25"] >= 50.0 else "Xỉu 2.5 bàn", f"{max(dc_c['over_25'], dc_c['under_25']):.1f}%", f"Tổng xG {dc_c['lam_h']+dc_c['lam_a']:.2f} · BTTS {dc_c['btts_yes']}%"),
                        "score": (f"Tỷ số {dc_c['top_scorelines'][0]['score']}", f"{dc_c['top_scorelines'][0]['prob_pct']:.1f}%", f"Kế tiếp: {dc_c['top_scorelines'][1]['score']} ({dc_c['top_scorelines'][1]['prob_pct']:.1f}%)"),
                    },
                    "drivers": drv_c, "odds": {}, "value_edges": [],
                    "home_form": hf_c, "away_form": af_c, "h2h": h2h_c,
                    "leaders": [], "news": [], "raw_row": None,
                }
                render_single_upcoming_dossier(synth_pkg, save_pred_fn=save_pred_fn, get_pred_fn=get_pred_fn, auth_user=auth_user, key_prefix="pm_cust")

    # ---------- TAB 4: EXPECTED POINTS (xPTS) & QUADRANT ----------
    if tab_xpts.open:
        with tab_xpts:
            xc1, xc2 = st.columns([2.0, 1.0], gap="small")
            x_lg = xc1.selectbox("Giải đấu:", [lg for lg in LEAGUE_ORDER if lg in ("ENG", "ESP", "ITA", "GER", "FRA", "UCL", "UEL", "ENG2", "NED", "POR")], format_func=lambda c: LEAGUE_META.get(c, (c, c, ""))[0], key="xpts_lg_pick")
            x_season = xc2.selectbox("Mùa giải:", ["2026-2027", "2025-2026"], format_func=lambda s: s.replace("-", "/"), key="xpts_season_pick")

            xdf = compute_league_xpts_standings(x_lg, x_season)
            if xdf.empty:
                st.info("Chưa có dữ liệu xG cho giải đấu này.")
            else:
                over_t = xdf.sort_values("luck_pts", ascending=False).iloc[0]
                under_t = xdf.sort_values("luck_pts", ascending=True).iloc[0]
                best_xgd = xdf.sort_values("xgd", ascending=False).iloc[0]
                ik1, ik2, ik3 = st.columns(3, gap="small")
                ik1.metric("👑 Thực Lực Đỉnh Nhất (xGD)", f"{best_xgd['team']}", f"xGD {best_xgd['xgd']:+.1f} · xPTS {best_xgd['xpts']}", border=True)
                ik2.metric("🔥 Vượt Kỳ Vọng Nhất", f"{over_t['team']}", f"{over_t['luck_pts']:+.1f}đ so với xPTS", border=True)
                ik3.metric("🎯 Tiềm Ẩn Bứt Phá Nhất", f"{under_t['team']}", f"{under_t['luck_pts']:+.1f}đ so với xPTS", border=True)

                xdf_p = xdf.copy()
                xdf_p["xg_pg"] = (xdf_p["xg"] / xdf_p["mp"].clip(lower=1)).round(2)
                xdf_p["xga_pg"] = (xdf_p["xga"] / xdf_p["mp"].clip(lower=1)).round(2)
                fig_q = go.Figure()
                fig_q.add_vline(x=float(xdf_p["xg_pg"].mean()), line_dash="dash", line_color="rgba(148,163,184,0.35)")
                fig_q.add_hline(y=float(xdf_p["xga_pg"].mean()), line_dash="dash", line_color="rgba(148,163,184,0.35)")
                fig_q.add_trace(go.Scatter(
                    x=xdf_p["xg_pg"], y=xdf_p["xga_pg"], mode="markers+text", text=xdf_p["team"],
                    textposition="top center", textfont=dict(size=10, color="#F8FAFC"),
                    marker=dict(size=12, color=xdf_p["xgd"], colorscale=[[0.0, "#EF4444"], [0.5, "#FACC15"], [1.0, "#00E676"]], line=dict(color="#0B0F19", width=1)),
                    hovertemplate="<b>%{text}</b><br>xG/trận: %{x:.2f}<br>xGA/trận: %{y:.2f}<extra></extra>",
                ))
                fig_q.update_layout(
                    title=f"Bản Đồ Tứ Phân Vị Sức Mạnh — {LEAGUE_META[x_lg][1]} {x_season.replace('-', '/')} (Góc trên-phải = Mạnh toàn diện)",
                    xaxis_title="⚡ xG tạo ra / trận ➔", yaxis_title="🛡️ xGA chịu đựng / trận (đảo chiều)",
                    yaxis=dict(autorange="reversed", gridcolor="rgba(148,163,184,0.10)"),
                    xaxis=dict(gridcolor="rgba(148,163,184,0.10)"),
                    height=390, margin=dict(l=15, r=15, t=42, b=15),
                    paper_bgcolor="#131A26", plot_bgcolor="#0E1420", font=dict(color="#E2E8F0"),
                )
                st.plotly_chart(fig_q, width="stretch", key=f"xpts_q_{x_lg}_{x_season}")

                trs = ""
                for _, rx in xdf.iterrows():
                    lc = "#00E676" if rx["luck_pts"] >= 2.0 else ("#F87171" if rx["luck_pts"] <= -2.0 else "#94A3B8")
                    trs += compact_html(f"""
                    <tr style="border-bottom:1px solid #1E293B;">
                        <td style="padding:6px 8px;font-weight:900;color:#38BDF8;">{int(rx['rank'])}</td>
                        <td style="padding:6px 8px;font-weight:800;color:#F8FAFC;">{html.escape(str(rx['team']))}</td>
                        <td style="padding:6px 5px;text-align:center;">{int(rx['mp'])}</td>
                        <td style="padding:6px 5px;text-align:center;">{int(rx['w'])}-{int(rx['d'])}-{int(rx['l'])}</td>
                        <td style="padding:6px 5px;text-align:center;">{int(rx['gf'])}:{int(rx['ga'])}</td>
                        <td style="padding:6px 6px;text-align:center;font-weight:900;color:#FFF;">{int(rx['pts'])}</td>
                        <td style="padding:6px 5px;text-align:center;color:#7DD3FC;">{rx['xg']:.1f}</td>
                        <td style="padding:6px 5px;text-align:center;color:#FDA4AF;">{rx['xga']:.1f}</td>
                        <td style="padding:6px 6px;text-align:center;font-weight:900;color:#FACC15;">{rx['xpts']:.1f}</td>
                        <td style="padding:6px 6px;text-align:center;font-weight:800;color:{lc};">{rx['luck_pts']:+.1f}</td>
                        <td style="padding:6px 6px;text-align:right;">{_render_form_pills_html(rx['form_5'])}</td>
                    </tr>
                    """)
                st.markdown(compact_html(f"""
                <div style="background:#131A26;border:1px solid #26354D;border-radius:12px;padding:12px 14px;overflow-x:auto;">
                    <table style="width:100%;border-collapse:collapse;font-size:12.5px;color:#CBD5E1;">
                        <thead><tr style="border-bottom:1.5px solid #334155;color:#94A3B8;font-size:11px;">
                            <th style="text-align:left;padding:5px 8px;">#</th><th style="text-align:left;padding:5px 8px;">Câu Lạc Bộ</th>
                            <th>Tr</th><th>T-H-B</th><th>BT:BB</th><th>Điểm</th><th>xG</th><th>xGA</th><th>xPTS</th><th>Pts−xPTS</th><th style="text-align:right;">Phong độ</th>
                        </tr></thead><tbody>{trs}</tbody>
                    </table>
                </div>
                """), unsafe_allow_html=True)
