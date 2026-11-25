"""Autonomous 24/7 ESPN -> Parquet/JSON data synchronization for NCKH TUD65."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata as _ud
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
PRO_DIR = BASE_DIR / "sample_data" / "pro_analytics"
PRO_DIR.mkdir(parents=True, exist_ok=True)

ESPN_CACHE_JSON = PRO_DIR / "espn_live_cache.json"
HIST_XG_PARQUET = PRO_DIR / "historical_matches_xg.parquet"
UPCOMING_PARQUET = PRO_DIR / "upcoming_fixtures.parquet"

VN_TZ = timezone(timedelta(hours=7))

ALL_LEAGUES: list[tuple[str, str, str]] = [
    ("ENG", "eng.1", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League"),
    ("ESP", "esp.1", "🇪🇸 Spanish La Liga"),
    ("ITA", "ita.1", "🇮🇹 Italian Serie A"),
    ("GER", "ger.1", "🇩🇪 German Bundesliga"),
    ("FRA", "fra.1", "🇫🇷 French Ligue 1"),
    ("UCL", "uefa.champions", "⚽ UEFA Champions League"),
    ("UEL", "uefa.europa", "🏆 UEFA Europa League"),
    ("UECL", "uefa.europa.conf", "🏅 UEFA Conference League"),
    ("ENG2", "eng.2", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 EFL Championship"),
    ("POR", "por.1", "🇵🇹 Liga Portugal"),
    ("NED", "ned.1", "🇳🇱 Eredivisie"),
    ("SCO", "sco.1", "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership"),
    ("TUR", "tur.1", "🇹🇷 Turkish Süper Lig"),
]

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
    "brighton and hove albion": "brighton",
    "nottingham forest": "nottm forest",
    "afc bournemouth": "bournemouth",
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
    "deportivo": "deportivo la coruna",
    "deportivo la coruna": "deportivo la coruna",
    "le havre ac": "le havre",
    "le havre": "le havre",
    "cologne": "koln",
    "1 fc koln": "koln",
    "fc koln": "koln",
    "paderborn 07": "paderborn",
    "sc paderborn 07": "paderborn",
}

_CLUB_MAP = {
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Arsenal FC": "Arsenal",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United FC": "Newcastle United",
    "Aston Villa FC": "Aston Villa",
    "Leicester City FC": "Leicester City",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Nottingham Forest FC": "Nottingham Forest",
    "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Atlético Madrid",
    "Sevilla FC": "Sevilla",
    "Real Betis Balompié": "Real Betis",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Villarreal CF": "Villarreal",
    "Athletic Club Bilbao": "Athletic Club",
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "AS Monaco",
    "LOSC Lille": "Lille",
    "FC Internazionale Milano": "Inter Milan",
    "Internazionale": "Inter Milan",
    "AC Milan": "AC Milan",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "AS Roma",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "Sporting Clube de Portugal": "Sporting CP",
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "AFC Ajax": "Ajax",
    "PSV Eindhoven": "PSV",
    "Feyenoord Rotterdam": "Feyenoord",
    "FK Bodø/Glimt": "Bodo/Glimt",
    "Bodø/Glimt": "Bodo/Glimt",
}


def clean_club_name(name: str) -> str:
    s = str(name or "").strip()
    if s in _CLUB_MAP:
        return _CLUB_MAP[s]
    s2 = re.sub(
        r"^(FC|FK|SK|NK|AC|SC|PFK|HŠK|GNK|KP|KKS|KS|BSC|US|SS|AS|ACF|SSC|RCD|CA|CD|UD|SV|VfL|VfB|TSG|OGC|AJ)\s+",
        "",
        s,
    )
    s2 = re.sub(r"\s+(FC|CF|KV|SK|FK|TC|SA|UBK|SCO|AFC|CFC|HSC|1907|1909|1910|1913|1893|1899|1846|1945)$", "", s2)
    s2 = s2.strip() or s
    return _CLUB_MAP.get(s2, s2)


def _norm_key(name: str) -> str:
    raw = str(name or "").translate(str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d"}))
    s = _ud.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    for tok in ("fc ", " fc", "afc ", " cf", " sc", " ac ", " ac", "sv ", "1. ", "sc ", "as ", "ss ", "us ", "rc ", "rcd ", "cd ", "ud ", "ogc "):
        s = s.replace(tok, " ")
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


def _team_alias_key(name: str) -> str:
    k = _norm_key(name)
    return _TEAM_ALIASES.get(k, k)


def _american_to_prob(ml) -> float | None:
    try:
        v = float(ml)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return (-v) / (-v + 100.0) if v < 0 else 100.0 / (v + 100.0)


def _fetch_json(url: str, timeout: float = 4.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NCKH-TUD65-AutoSync/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_league_standings(code: str, slug: str, comp_label: str) -> tuple[str, dict | None]:
    url = f"https://site.api.espn.com/apis/v2/sports/soccer/{slug}/standings"
    try:
        st_raw = _fetch_json(url, timeout=5.0)
        ch = st_raw.get("children") or []
        rows = []
        if ch:
            for ent in (ch[0].get("standings", {}).get("entries") or []):
                t_info = ent.get("team") or {}
                s_map = {s.get("name"): s.get("displayValue") for s in (ent.get("stats") or [])}
                rows.append(
                    {
                        "rank": int(float(s_map.get("rank") or len(rows) + 1)),
                        "team": clean_club_name(t_info.get("displayName", "")),
                        "team_raw": str(t_info.get("displayName") or ""),
                        "logo": str((t_info.get("logos") or [{}])[0].get("href") or ""),
                        "mp": int(float(s_map.get("gamesPlayed") or 0)),
                        "w": int(float(s_map.get("wins") or 0)),
                        "d": int(float(s_map.get("ties") or 0)),
                        "l": int(float(s_map.get("losses") or 0)),
                        "gf": int(float(s_map.get("pointsFor") or 0)),
                        "ga": int(float(s_map.get("pointsAgainst") or 0)),
                        "gd": int(float(str(s_map.get("pointDifferential") or "0").replace("+", "") or 0)),
                        "pts": int(float(s_map.get("points") or 0)),
                    }
                )
            rows.sort(key=lambda r: (r["rank"], -r["pts"], -r["gd"]))
        return code, {"league_name": comp_label, "slug": slug, "rows": rows}
    except Exception:
        return code, None


def fetch_scoreboard_day(code: str, slug: str, comp_label: str, dt_yyyymmdd: str) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dt_yyyymmdd}"
    try:
        sb = _fetch_json(url, timeout=4.0)
    except Exception:
        return []

    rows = []
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

            h_team = home_c.get("team") or {}
            a_team = away_c.get("team") or {}
            h_name = clean_club_name(h_team.get("displayName", ""))
            a_name = clean_club_name(a_team.get("displayName", ""))
            if not h_name or not a_name:
                continue

            h_logos = h_team.get("logos") or []
            a_logos = a_team.get("logos") or []
            h_logo = str(h_team.get("logo") or (h_logos[0].get("href") if h_logos and isinstance(h_logos[0], dict) else "") or "")
            a_logo = str(a_team.get("logo") or (a_logos[0].get("href") if a_logos and isinstance(a_logos[0], dict) else "") or "")

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
            if kickoff_utc:
                try:
                    dt_u = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
                    dt_v = dt_u.astimezone(VN_TZ)
                    vn_date = dt_v.strftime("%Y-%m-%d")
                    vn_time = dt_v.strftime("%H:%M")
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
                        "provider": str((od0.get("provider") or {}).get("name") or "DraftKings"),
                        "details": str(od0.get("details") or ""),
                        "over_under": od0.get("overUnder"),
                        "spread": od0.get("spread"),
                        "home_ml": h_ml,
                        "draw_ml": d_ml,
                        "away_ml": a_ml,
                        "mkt_prob_H": round(ph / tot_p * 100, 1),
                        "mkt_prob_D": round(pd_ / tot_p * 100, 1),
                        "mkt_prob_A": round(pa / tot_p * 100, 1),
                    }

            rows.append(
                {
                    "espn_event_id": eid,
                    "league_code": code,
                    "espn_slug": slug,
                    "competition": comp_label,
                    "kickoff_utc": kickoff_utc,
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
                    "home_form": str(home_c.get("form") or ""),
                    "away_form": str(away_c.get("form") or ""),
                    "venue": f"{venue} ({city})" if venue and city and city not in venue else venue,
                    "odds": odds_info,
                }
            )
        except Exception:
            continue
    return rows


def _infer_season(match_date_str: str) -> str:
    try:
        dt = date.fromisoformat(match_date_str[:10])
        if dt.month >= 7:
            return f"{dt.year}-{dt.year + 1}"
        return f"{dt.year - 1}-{dt.year}"
    except Exception:
        return "2026-2027"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(f".parquet.tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_semantic_digest(standings: dict, fixtures: list[dict]) -> str:
    """Stable hash of standings & fixtures ignoring fetch ordering or wall-clock sync time."""
    compact_fx = []
    for f in sorted(fixtures, key=lambda x: str(x.get("espn_event_id") or "")):
        compact_fx.append(
            [
                f.get("espn_event_id"),
                f.get("league_code"),
                f.get("state"),
                f.get("home_score"),
                f.get("away_score"),
                f.get("status_detail"),
                f.get("vn_date"),
                f.get("vn_time"),
                f.get("home_team"),
                f.get("away_team"),
            ]
        )
    blob = json.dumps({"standings": standings, "fixtures": compact_fx}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_sync() -> dict:
    files_changed = 0
    now_utc = datetime.now(timezone.utc)
    now_vn = now_utc.astimezone(VN_TZ)
    today_vn_str = now_vn.strftime("%Y-%m-%d")

    existing_cache = {"updated_utc": "", "standings": {}, "live_fixtures": []}
    if ESPN_CACHE_JSON.is_file():
        try:
            with open(ESPN_CACHE_JSON, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing_cache = loaded
        except Exception:
            pass

    hist_df = pd.read_parquet(HIST_XG_PARQUET) if HIST_XG_PARQUET.is_file() else pd.DataFrame()
    hist_max_date = str(hist_df["match_date"].astype(str).str.slice(0, 10).max()) if not hist_df.empty else ""

    # Determine scoreboard date span (dynamic window catching every gap up to 14 days back + 10 days ahead)
    min_start = now_utc.date() - timedelta(days=14)
    default_start = now_utc.date() - timedelta(days=3)
    if hist_max_date:
        try:
            hd = date.fromisoformat(hist_max_date) - timedelta(days=1)
            default_start = min(default_start, hd)
        except Exception:
            pass
    start_d = max(min_start, default_start)
    end_d = now_utc.date() + timedelta(days=10)

    date_strs = []
    cur_d = start_d
    while cur_d <= end_d:
        date_strs.append(cur_d.strftime("%Y%m%d"))
        cur_d += timedelta(days=1)

    # 1. Parallel standings fetch across all 13 leagues
    new_standings = dict(existing_cache.get("standings") or {})
    with ThreadPoolExecutor(max_workers=13) as pool:
        futs = [pool.submit(fetch_league_standings, c, s, l) for c, s, l in ALL_LEAGUES]
        for fut in as_completed(futs):
            code, data = fut.result()
            if data is not None and data.get("rows"):
                new_standings[code] = data

    # 2. Parallel scoreboards fetch
    sb_tasks = [(c, s, l, d) for c, s, l in ALL_LEAGUES for d in date_strs]
    fresh_by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(fetch_scoreboard_day, c, s, l, d) for c, s, l, d in sb_tasks]
        for fut in as_completed(futs):
            for ev in fut.result():
                fresh_by_id[str(ev["espn_event_id"])] = ev

    # Merge with existing persisted finished/upcoming fixtures in cache
    merged_by_id: dict[str, dict] = {}
    cutoff_keep = (now_vn.date() - timedelta(days=270)).strftime("%Y-%m-%d")
    for old in (existing_cache.get("live_fixtures") or []):
        if not isinstance(old, dict):
            continue
        eid = str(old.get("espn_event_id") or "").strip()
        od = str(old.get("vn_date") or old.get("match_date") or "")[:10]
        if eid and (not od or od >= cutoff_keep):
            merged_by_id[eid] = old

    for eid, ev in fresh_by_id.items():
        merged_by_id[eid] = ev

    merged_fixtures = sorted(
        merged_by_id.values(),
        key=lambda x: (str(x.get("vn_date") or x.get("match_date") or ""), str(x.get("espn_event_id") or "")),
    )

    # 3. Append newly finished ESPN matches into historical_matches_xg.parquet
    added_hist = 0
    elo_map: dict[str, float] = {}
    canon_by_league: dict[tuple[str, str], str] = {}
    canon_global: dict[str, str] = {}
    existing_mids: set[str] = set()
    existing_sigs: dict[tuple[str, str, str], list[date]] = {}
    finished_sigs: list[tuple[str, str, date]] = []

    if not hist_df.empty:
        orig_cols = list(hist_df.columns)
        orig_dtypes = {c: hist_df[c].dtype for c in orig_cols}
        h_sorted = hist_df.copy()
        h_sorted["_md"] = h_sorted["match_date"].astype(str).str.slice(0, 10)
        h_sorted = h_sorted.sort_values("_md")

        for r in h_sorted.itertuples(index=False):
            lg = str(getattr(r, "league", "") or "")
            ht = str(getattr(r, "home_team", "") or "")
            at = str(getattr(r, "away_team", "") or "")
            mid = str(getattr(r, "match_id", "") or "")
            md_s = str(getattr(r, "_md", "") or "")
            season_s = str(getattr(r, "season", "") or "")

            hk, ak = _team_alias_key(ht), _team_alias_key(at)
            if mid:
                existing_mids.add(mid)
            if season_s >= "2025-2026":
                if hk:
                    canon_by_league.setdefault((lg, hk), ht)
                    canon_global.setdefault(hk, ht)
                if ak:
                    canon_by_league.setdefault((lg, ak), at)
                    canon_global.setdefault(ak, at)

            he = getattr(r, "home_elo", None)
            ae = getattr(r, "away_elo", None)
            if he is not None and pd.notna(he) and hk:
                elo_map[hk] = float(he)
            if ae is not None and pd.notna(ae) and ak:
                elo_map[ak] = float(ae)

            try:
                dt_o = date.fromisoformat(md_s)
                existing_sigs.setdefault((lg, hk, ak), []).append(dt_o)
                if season_s >= "2026-2027":
                    finished_sigs.append((hk, ak, dt_o))
            except Exception:
                pass

        # Chronologically collect post fixtures not yet recorded
        post_cands = [
            ev
            for ev in merged_fixtures
            if str(ev.get("state") or "").lower() == "post"
            and ev.get("home_score") is not None
            and ev.get("away_score") is not None
        ]
        post_cands.sort(key=lambda x: str(x.get("vn_date") or x.get("match_date") or ""))

        new_rows = []
        for ev in post_cands:
            lg = str(ev.get("league_code") or "")
            if not lg:
                continue
            eid = str(ev.get("espn_event_id") or "")
            mid_val = f"espn_{eid}"
            if mid_val in existing_mids or eid in existing_mids:
                continue

            md_s = str(ev.get("vn_date") or ev.get("match_date") or "")[:10]
            try:
                dt_ev = date.fromisoformat(md_s)
            except Exception:
                continue

            ht_raw = str(ev.get("home_team") or "")
            at_raw = str(ev.get("away_team") or "")
            hk, ak = _team_alias_key(ht_raw), _team_alias_key(at_raw)
            if not hk or not ak:
                continue

            # Deduplicate across both exact league and +-2 day rollover
            dup_dates = existing_sigs.get((lg, hk, ak), [])
            if any(abs((dt_ev - d0).days) <= 2 for d0 in dup_dates):
                finished_sigs.append((hk, ak, dt_ev))
                continue

            ht_canon = canon_by_league.get((lg, hk)) or canon_global.get(hk) or ht_raw
            at_canon = canon_by_league.get((lg, ak)) or canon_global.get(ak) or at_raw
            hg = int(ev["home_score"])
            ag = int(ev["away_score"])

            pre_h_elo = float(elo_map.get(hk, 1550.0))
            pre_a_elo = float(elo_map.get(ak, 1550.0))

            # Calibrated xG & Elo update
            elo_diff_adj = (pre_h_elo + 65.0) - pre_a_elo
            exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff_adj / 400.0))
            act_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            mult = 1.0 + 0.18 * min(abs(hg - ag), 4)
            delta_elo = 20.0 * mult * (act_h - exp_h)
            elo_map[hk] = round(pre_h_elo + delta_elo, 2)
            elo_map[ak] = round(pre_a_elo - delta_elo, 2)

            prior_h_xg = float(np.clip(1.42 + (elo_diff_adj / 650.0), 0.45, 2.85))
            prior_a_xg = float(np.clip(1.14 - (elo_diff_adj / 650.0), 0.40, 2.65))
            cal_h_xg = round(float(np.clip(0.68 * hg + 0.32 * prior_h_xg, 0.15, 4.80)), 2)
            cal_a_xg = round(float(np.clip(0.68 * ag + 0.32 * prior_a_xg, 0.15, 4.80)), 2)

            season_val = _infer_season(md_s)
            row_dict = {col: None for col in orig_cols}
            row_dict.update(
                {
                    "match_id": mid_val,
                    "match_date": md_s,
                    "league": lg,
                    "season": season_val,
                    "home_team": ht_canon,
                    "away_team": at_canon,
                    "home_goals": float(hg),
                    "away_goals": float(ag),
                    "home_xg": cal_h_xg,
                    "away_xg": cal_a_xg,
                    "home_elo": round(pre_h_elo, 2),
                    "away_elo": round(pre_a_elo, 2),
                    "home_sum_panna": 0.0,
                    "away_sum_panna": 0.0,
                    "home_score_ht": "0",
                    "away_score_ht": "0",
                }
            )
            new_rows.append(row_dict)
            existing_mids.add(mid_val)
            existing_sigs.setdefault((lg, hk, ak), []).append(dt_ev)
            finished_sigs.append((hk, ak, dt_ev))

        if new_rows:
            df_add = pd.DataFrame(new_rows, columns=orig_cols)
            for c, dt in orig_dtypes.items():
                try:
                    df_add[c] = df_add[c].astype(dt)
                except Exception:
                    pass
            hist_updated = pd.concat([hist_df, df_add], ignore_index=True)
            _atomic_write_parquet(HIST_XG_PARQUET, hist_updated)
            added_hist = len(new_rows)
            files_changed += 1

    # Also add post signatures from fresh/cached ESPN events so upcoming_fixtures removes them cleanly
    for ev in merged_fixtures:
        if str(ev.get("state") or "").lower() == "post":
            md_s = str(ev.get("vn_date") or ev.get("match_date") or "")[:10]
            try:
                finished_sigs.append((_team_alias_key(ev.get("home_team", "")), _team_alias_key(ev.get("away_team", "")), date.fromisoformat(md_s)))
            except Exception:
                pass

    # Build quick lookup index for upcoming filtering
    fin_by_pair: dict[tuple[str, str], list[date]] = {}
    for hk, ak, d0 in finished_sigs:
        fin_by_pair.setdefault((hk, ak), []).append(d0)

    # 4. Clean completed & stale fixtures from upcoming_fixtures.parquet + refresh Elos
    removed_upcoming = 0
    if UPCOMING_PARQUET.is_file():
        up_df = pd.read_parquet(UPCOMING_PARQUET)
        if not up_df.empty:
            up_orig_cols = list(up_df.columns)
            up_orig_dtypes = {c: up_df[c].dtype for c in up_orig_cols}
            stale_cutoff = (now_vn.date() - timedelta(days=1)).strftime("%Y-%m-%d")
            keep_mask = []
            elo_dirty = False
            new_h_elos = []
            new_a_elos = []

            for r in up_df.itertuples(index=False):
                md_u = str(getattr(r, "match_date", "") or "")[:10]
                ht_u = str(getattr(r, "home_team", "") or "")
                at_u = str(getattr(r, "away_team", "") or "")
                hk_u, ak_u = _team_alias_key(ht_u), _team_alias_key(at_u)

                is_done = False
                try:
                    dt_u = date.fromisoformat(md_u)
                    if any(abs((dt_u - d0).days) <= 2 for d0 in fin_by_pair.get((hk_u, ak_u), [])):
                        is_done = True
                except Exception:
                    pass
                if md_u and md_u < stale_cutoff:
                    is_done = True

                keep_mask.append(not is_done)
                old_he = float(getattr(r, "home_elo", 1550.0) or 1550.0)
                old_ae = float(getattr(r, "away_elo", 1550.0) or 1550.0)
                upd_he = float(elo_map.get(hk_u, old_he))
                upd_ae = float(elo_map.get(ak_u, old_ae))
                if not is_done and (abs(upd_he - old_he) >= 0.5 or abs(upd_ae - old_ae) >= 0.5):
                    elo_dirty = True
                new_h_elos.append(upd_he)
                new_a_elos.append(upd_ae)

            up_new = up_df.copy()
            up_new["home_elo"] = new_h_elos
            up_new["away_elo"] = new_a_elos
            up_new = up_new[keep_mask].reset_index(drop=True)
            removed_upcoming = len(up_df) - len(up_new)

            if removed_upcoming > 0 or elo_dirty:
                for c, dt in up_orig_dtypes.items():
                    try:
                        up_new[c] = up_new[c].astype(dt)
                    except Exception:
                        pass
                _atomic_write_parquet(UPCOMING_PARQUET, up_new[up_orig_cols])
                files_changed += 1

    # 5. Semantic-diff check before updating espn_live_cache.json
    old_digest = _json_semantic_digest(
        existing_cache.get("standings") or {},
        existing_cache.get("live_fixtures") or [],
    )
    new_digest = _json_semantic_digest(new_standings, merged_fixtures)
    if new_digest != old_digest or not ESPN_CACHE_JSON.is_file():
        payload = {
            "updated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "standings": new_standings,
            "live_fixtures": merged_fixtures,
        }
        _atomic_write_json(ESPN_CACHE_JSON, payload)
        files_changed += 1

    summary = {
        "new_hist_matches": added_hist,
        "removed_upcoming": removed_upcoming,
        "standings_leagues": len(new_standings),
        "live_fixtures_total": len(merged_fixtures),
        "files_changed": files_changed,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_sync()
