"""
Trung Tâm Lịch Đấu Thời Gian Thực & Trận Chuẩn Bị Diễn Ra (Real-Time Daily Match Center)
- Tự động đồng bộ tỷ số trực tiếp, đồng hồ trận đấu & lịch thi đấu GMT+7 (giờ Việt Nam) từ ESPN Live API
- Bộ nhớ đệm Stale-While-Revalidate dùng chung toàn server: nhiều người xem cùng lúc không bị lag hay spam API
- Kết hợp 1,928 trận sắp đá có sẵn dự báo AI Opta ML + Dixon-Coles + DraftKings Odds
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import html
import json
import os
import threading
import time
import urllib.request

import pandas as pd
import streamlit as st

from build_pro_analytics_store import ESPN_LEAGUES, clean_club_name
from prematch_research import (
    CATALOG_CATEGORY_MAP,
    ESPN_CACHE_JSON,
    LEAGUE_META,
    LEAGUE_ORDER,
    _american_to_prob,
    _safe_float,
    _team_alias_key,
    compact_html,
    load_espn_live_cache,
    load_historical_xg_df,
    load_upcoming_fixtures_df,
)
from team_logos_store import get_team_logo_url

VN_TZ = timezone(timedelta(hours=7))

# Process-shared stale-while-revalidate state across all concurrent viewers
_BOARD_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
_BOARD_CACHE: dict = {}
_REFRESH_IN_PROGRESS: bool = False
_LAST_FORCE_MONO: float = 0.0
_CACHE_TTL_SEC = 180.0
_FORCE_COOLDOWN_SEC = 15.0

# Process-shared ESPN live match summary cache (for inline Live Match Viewer)
_SUMMARY_LOCK = threading.Lock()
_SUMMARY_CACHE: dict[tuple[str, str], dict] = {}
_SUMMARY_FORCE_MONO: dict[tuple[str, str], float] = {}
_SUMMARY_LIVE_TTL_SEC = 20.0
_SUMMARY_STATIC_TTL_SEC = 180.0
_SUMMARY_FORCE_COOLDOWN_SEC = 8.0

_EXTENDED_LEAGUE_ORDER = list(LEAGUE_ORDER) + ["INTL_FR"]
_VI_WEEKDAYS = ["Th 2", "Th 3", "Th 4", "Th 5", "Th 6", "Th 7", "CN"]


def _parse_kickoff_dt(kickoff_utc: str):
    if not kickoff_utc or "T" not in str(kickoff_utc):
        return None
    try:
        s = str(kickoff_utc).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt_utc = datetime.fromisoformat(s)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc
    except Exception:
        return None


def _normalize_live_entry(item: dict) -> dict:
    """Ensure any cached or freshly fetched ESPN entry has vn_date, vn_time, kickoff_ts, state, and scores."""
    out = dict(item)
    kickoff_utc = str(out.get("kickoff_utc") or "")
    dt_utc = _parse_kickoff_dt(kickoff_utc)
    if dt_utc is not None:
        dt_vn = dt_utc.astimezone(VN_TZ)
        out["vn_date"] = out.get("vn_date") or dt_vn.strftime("%Y-%m-%d")
        out["vn_time"] = out.get("vn_time") or dt_vn.strftime("%H:%M")
        out["kickoff_ts"] = float(out.get("kickoff_ts") or dt_utc.timestamp())
    else:
        out["vn_date"] = str(out.get("vn_date") or out.get("match_date") or "")[:10]
        out["vn_time"] = str(out.get("vn_time") or "")
        out["kickoff_ts"] = float(out.get("kickoff_ts") or 0.0)

    raw_state = str(out.get("state") or "").lower()
    st_name = str(out.get("status") or "").upper()
    if raw_state not in ("in", "pre", "post"):
        if "FINAL" in st_name or "FULL_TIME" in st_name or "AFTER_EXTRA" in st_name:
            raw_state = "post"
        elif any(k in st_name for k in ("IN_PROGRESS", "HALFTIME", "FIRST_HALF", "SECOND_HALF", "EXTRA_TIME", "PENALTIES")):
            raw_state = "in"
        else:
            raw_state = "pre"
    out["state"] = raw_state

    if not out.get("status_detail"):
        if raw_state == "post":
            out["status_detail"] = "FT"
        elif raw_state == "in":
            out["status_detail"] = "LIVE"
        else:
            out["status_detail"] = out["vn_time"] or ("Hoãn" if "POSTPONED" in st_name else "Theo lịch")
    out["home_logo"] = get_team_logo_url(out.get("home_team", ""), fallback_url=out.get("home_logo", ""))
    out["away_logo"] = get_team_logo_url(out.get("away_team", ""), fallback_url=out.get("away_logo", ""))
    return out


def _fetch_single_espn_scoreboard(task: tuple, timeout: float) -> list[dict]:
    code, slug, comp_label, dt_str = task
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dt_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        sb = json.loads(resp.read().decode("utf-8"))

    rows = []
    for ev in sb.get("events", []):
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
            dt_utc = _parse_kickoff_dt(kickoff_utc)
            if dt_utc is not None:
                dt_vn = dt_utc.astimezone(VN_TZ)
                vn_date = dt_vn.strftime("%Y-%m-%d")
                vn_time = dt_vn.strftime("%H:%M")
                kickoff_ts = dt_utc.timestamp()
            else:
                vn_date = kickoff_utc[:10] or f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
                vn_time = ""
                kickoff_ts = 0.0

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

            rows.append({
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
    return rows


def fetch_espn_scoreboard_window(days_back: int = 1, days_ahead: int = 7, timeout: float = 2.5) -> dict:
    """Fetch ESPN scoreboards in parallel across Top 7 competitions (live, upcoming, and recent results)."""
    now_utc = datetime.now(timezone.utc)
    dates_to_fetch = [(now_utc + timedelta(days=d)).strftime("%Y%m%d") for d in range(-days_back, days_ahead + 1)]
    tasks = [
        (code, slug, comp_label, dt_s)
        for code, slug, comp_label in ESPN_LEAGUES
        for dt_s in dates_to_fetch
    ]

    by_id: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(_fetch_single_espn_scoreboard, t, timeout) for t in tasks]
        for fut in as_completed(futs):
            try:
                for item in fut.result():
                    by_id[item["espn_event_id"]] = item
            except Exception:
                continue

    fixtures = list(by_id.values())
    now_vn = datetime.now(VN_TZ)
    return {
        "updated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_vn": now_vn.strftime("%H:%M:%S %d/%m"),
        "fetched_mono": time.monotonic(),
        "live_fixtures": fixtures,
    }


def _read_disk_cache_normalized() -> dict:
    if not os.path.isfile(ESPN_CACHE_JSON):
        return {"updated_utc": "", "updated_vn": "", "live_fixtures": [], "disk_age": 1e9}
    try:
        with open(ESPN_CACHE_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"updated_utc": "", "updated_vn": "", "live_fixtures": [], "disk_age": 1e9}

    upd_utc = str(raw.get("updated_utc") or "")
    age_s = 1e9
    upd_vn = ""
    dt_upd = _parse_kickoff_dt(upd_utc)
    if dt_upd is not None:
        age_s = max(0.0, (datetime.now(timezone.utc) - dt_upd).total_seconds())
        upd_vn = dt_upd.astimezone(VN_TZ).strftime("%H:%M:%S %d/%m")

    items = [_normalize_live_entry(x) for x in (raw.get("live_fixtures") or [])]
    return {
        "updated_utc": upd_utc,
        "updated_vn": upd_vn or upd_utc,
        "live_fixtures": items,
        "disk_age": age_s,
    }


def _persist_disk_cache(live_fixtures: list, updated_utc: str):
    if not live_fixtures:
        return
    try:
        existing_standings = {}
        if os.path.isfile(ESPN_CACHE_JSON):
            try:
                with open(ESPN_CACHE_JSON, "r", encoding="utf-8") as f:
                    existing_standings = json.load(f).get("standings") or {}
            except Exception:
                existing_standings = {}
        payload = {
            "updated_utc": updated_utc,
            "standings": existing_standings,
            "live_fixtures": live_fixtures,
        }
        tmp_p = ESPN_CACHE_JSON + ".tmp"
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_p, ESPN_CACHE_JSON)
        try:
            load_espn_live_cache.clear()
        except Exception:
            pass
    except Exception:
        pass


def _spawn_background_refresh():
    global _REFRESH_IN_PROGRESS
    with _BOARD_LOCK:
        if _REFRESH_IN_PROGRESS:
            return
        _REFRESH_IN_PROGRESS = True

    def _worker():
        global _REFRESH_IN_PROGRESS
        try:
            fresh = fetch_espn_scoreboard_window(days_back=1, days_ahead=7, timeout=2.5)
            if fresh.get("live_fixtures"):
                with _BOARD_LOCK:
                    _BOARD_CACHE.clear()
                    _BOARD_CACHE.update(fresh)
                _persist_disk_cache(fresh["live_fixtures"], fresh["updated_utc"])
        except Exception:
            pass
        finally:
            with _BOARD_LOCK:
                _REFRESH_IN_PROGRESS = False

    t = threading.Thread(target=_worker, name="DailyRealtimeESPNRefresh", daemon=True)
    t.start()


def get_realtime_board_data(force_refresh: bool = False) -> dict:
    """Process-shared stale-while-revalidate live ESPN board provider (safe for concurrent users)."""
    global _LAST_FORCE_MONO
    now_m = time.monotonic()

    if force_refresh:
        with _BOARD_LOCK:
            if _BOARD_CACHE and (now_m - _LAST_FORCE_MONO) < _FORCE_COOLDOWN_SEC:
                res = dict(_BOARD_CACHE)
                res["cooldown_wait"] = int(_FORCE_COOLDOWN_SEC - (now_m - _LAST_FORCE_MONO))
                return res
            _LAST_FORCE_MONO = now_m
        with _FETCH_LOCK:
            fresh = fetch_espn_scoreboard_window(days_back=1, days_ahead=7, timeout=2.5)
            if fresh.get("live_fixtures"):
                with _BOARD_LOCK:
                    _BOARD_CACHE.clear()
                    _BOARD_CACHE.update(fresh)
                _persist_disk_cache(fresh["live_fixtures"], fresh["updated_utc"])
                return dict(_BOARD_CACHE)

    with _BOARD_LOCK:
        cached = dict(_BOARD_CACHE) if _BOARD_CACHE else None

    if cached is not None:
        age = now_m - float(cached.get("fetched_mono", 0.0))
        if age > _CACHE_TTL_SEC:
            _spawn_background_refresh()
        return cached

    # Cold start: only one request hits network, other concurrent callers wait cleanly on _FETCH_LOCK
    with _FETCH_LOCK:
        with _BOARD_LOCK:
            if _BOARD_CACHE:
                return dict(_BOARD_CACHE)

        disk = _read_disk_cache_normalized()
        if disk["live_fixtures"] and disk["disk_age"] <= _CACHE_TTL_SEC:
            cold_from_disk = {
                "updated_utc": disk["updated_utc"],
                "updated_vn": disk["updated_vn"],
                "fetched_mono": now_m - disk["disk_age"],
                "live_fixtures": disk["live_fixtures"],
            }
            with _BOARD_LOCK:
                _BOARD_CACHE.clear()
                _BOARD_CACHE.update(cold_from_disk)
            return dict(_BOARD_CACHE)

        fresh = fetch_espn_scoreboard_window(days_back=1, days_ahead=7, timeout=2.5)
        if fresh.get("live_fixtures"):
            with _BOARD_LOCK:
                _BOARD_CACHE.clear()
                _BOARD_CACHE.update(fresh)
            _persist_disk_cache(fresh["live_fixtures"], fresh["updated_utc"])
            return dict(_BOARD_CACHE)

        # Graceful fallback to disk snapshot when offline/ESPN unreachable
        fallback = {
            "updated_utc": disk["updated_utc"],
            "updated_vn": disk["updated_vn"] or "Dữ liệu cục bộ",
            "fetched_mono": now_m,
            "live_fixtures": disk["live_fixtures"],
        }
        with _BOARD_LOCK:
            _BOARD_CACHE.clear()
            _BOARD_CACHE.update(fallback)
        return dict(_BOARD_CACHE)


def _safe_hex_color(raw: str, default: str) -> str:
    s = str(raw or "").strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s}"
    if len(s) == 3 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s[0]*2}{s[1]*2}{s[2]*2}"
    return default


def _extract_summary_team_info(comp_entry: dict, is_home: bool) -> dict:
    t_obj = (comp_entry or {}).get("team") or {}
    logos = t_obj.get("logos") or []
    logo = str(t_obj.get("logo") or (logos[0].get("href") if logos and isinstance(logos[0], dict) else "") or "")
    rec_list = (comp_entry or {}).get("record") or []
    rec_txt = ""
    pts_txt = ""
    for r in rec_list:
        if not isinstance(r, dict):
            continue
        rtype = str(r.get("type") or "").lower()
        if rtype == "total":
            rec_txt = str(r.get("displayValue") or r.get("summary") or "")
        elif rtype == "points":
            pts_txt = str(r.get("displayValue") or r.get("summary") or "")
    default_col = "#38BDF8" if is_home else "#F43F5E"
    return {
        "id": str(t_obj.get("id") or ""),
        "name": clean_club_name(t_obj.get("displayName", "")) or ("Home" if is_home else "Away"),
        "logo": logo,
        "color": _safe_hex_color(t_obj.get("color", ""), default_col),
        "score": None,
        "record": rec_txt,
        "points": pts_txt,
    }


def _parse_boxscore_stats(teams_bs: list, home_id: str, away_id: str) -> list[tuple]:
    if not isinstance(teams_bs, list) or len(teams_bs) < 2:
        return []
    h_bs, a_bs = None, None
    for tb in teams_bs:
        if not isinstance(tb, dict):
            continue
        ha = str(tb.get("homeAway") or "").lower()
        tid = str((tb.get("team") or {}).get("id") or "")
        if ha == "home" or (home_id and tid == home_id):
            h_bs = tb
        elif ha == "away" or (away_id and tid == away_id):
            a_bs = tb
    if h_bs is None or a_bs is None:
        h_bs, a_bs = teams_bs[0], teams_bs[1]

    def _stat_map(tb: dict) -> dict[str, tuple[float | None, str]]:
        out = {}
        for s in (tb.get("statistics") or []):
            if not isinstance(s, dict) or not s.get("name"):
                continue
            disp = str(s.get("displayValue") or "").strip()
            try:
                num = float(disp.replace("%", ""))
            except (TypeError, ValueError):
                num = None
            out[str(s["name"])] = (num, disp)
        return out

    hm = _stat_map(h_bs)
    am = _stat_map(a_bs)
    if not hm and not am:
        return []

    def _num(mp: dict, key: str, default: float = 0.0) -> float:
        v = mp.get(key)
        return v[0] if v and v[0] is not None else default

    def _has_any(*keys: str) -> bool:
        return any(k in hm or k in am for k in keys)

    def _pct_norm(mp: dict, key: str) -> float:
        v = _num(mp, key, 0.0)
        if 0.0 < v <= 1.01 and key != "possessionPct":
            return v * 100.0
        return v

    rows: list[tuple] = []
    if _has_any("possessionPct"):
        hv, av = _num(hm, "possessionPct"), _num(am, "possessionPct")
        rows.append(("possessionPct", "Kiểm soát bóng (%)", f"{hv:.1f}%", f"{av:.1f}%", hv, av))
    if _has_any("totalShots"):
        hv, av = int(round(_num(hm, "totalShots"))), int(round(_num(am, "totalShots")))
        rows.append(("totalShots", "Tổng cú sút", str(hv), str(av), float(hv), float(av)))
    if _has_any("shotsOnTarget"):
        hv, av = int(round(_num(hm, "shotsOnTarget"))), int(round(_num(am, "shotsOnTarget")))
        rows.append(("shotsOnTarget", "Sút trúng đích", str(hv), str(av), float(hv), float(av)))
    if _has_any("accuratePasses", "totalPasses"):
        h_acc, h_tot = int(round(_num(hm, "accuratePasses"))), int(round(_num(hm, "totalPasses")))
        a_acc, a_tot = int(round(_num(am, "accuratePasses"))), int(round(_num(am, "totalPasses")))
        h_pct = (h_acc / h_tot * 100.0) if h_tot > 0 else _pct_norm(hm, "passPct")
        a_pct = (a_acc / a_tot * 100.0) if a_tot > 0 else _pct_norm(am, "passPct")
        h_str = f"{h_acc}/{h_tot} ({h_pct:.0f}%)" if h_tot > 0 else str(h_acc)
        a_str = f"{a_acc}/{a_tot} ({a_pct:.0f}%)" if a_tot > 0 else str(a_acc)
        rows.append(("accuratePasses", "Chuyền chính xác", h_str, a_str, float(h_acc), float(a_acc)))
    if _has_any("passPct", "accuratePasses"):
        h_tot, a_tot = _num(hm, "totalPasses"), _num(am, "totalPasses")
        h_p = (_num(hm, "accuratePasses") / h_tot * 100.0) if h_tot > 0 else _pct_norm(hm, "passPct")
        a_p = (_num(am, "accuratePasses") / a_tot * 100.0) if a_tot > 0 else _pct_norm(am, "passPct")
        rows.append(("passPct", "Tỷ lệ chuyền (%)", f"{h_p:.1f}%", f"{a_p:.1f}%", round(h_p, 1), round(a_p, 1)))

    simple_map = [
        ("wonCorners", "Phạt góc"),
        ("saves", "Thủ môn cứu thua"),
        ("totalTackles", "Tắc bóng"),
        ("interceptions", "Cắt bóng"),
        ("totalClearance", "Phá bóng"),
        ("foulsCommitted", "Phạm lỗi"),
        ("offsides", "Việt vị"),
        ("yellowCards", "Thẻ vàng"),
        ("redCards", "Thẻ đỏ"),
    ]
    for k, lbl in simple_map:
        if _has_any(k):
            hv, av = int(round(_num(hm, k))), int(round(_num(am, k)))
            rows.append((k, lbl, str(hv), str(av), float(hv), float(av)))
    return rows


def _parse_espn_roster(r_obj: dict, is_home: bool) -> dict:
    t_obj = (r_obj or {}).get("team") or {}
    unif = (r_obj or {}).get("uniform") or {}
    kit_default = "#38BDF8" if is_home else "#F43F5E"
    kit_color = _safe_hex_color(unif.get("color", ""), kit_default)
    alt_color = _safe_hex_color(unif.get("alternateColor", ""), "#FFFFFF")

    starters, bench = [], []
    for p in ((r_obj or {}).get("roster") or []):
        if not isinstance(p, dict):
            continue
        ath = p.get("athlete") or {}
        pos_obj = p.get("position") or {}
        st_map = {}
        for st_item in (p.get("stats") or []):
            if isinstance(st_item, dict) and st_item.get("name"):
                st_map[str(st_item["name"])] = _safe_float(st_item.get("value"), 0.0)
        entry = {
            "name": str(ath.get("displayName") or ath.get("shortName") or ""),
            "short_name": str(ath.get("shortName") or ath.get("displayName") or ""),
            "jersey": str(p.get("jersey") or ""),
            "pos": str(pos_obj.get("abbreviation") or pos_obj.get("name") or ""),
            "fp": str(p.get("formationPlace") or ""),
            "starter": bool(p.get("starter")),
            "subbed_in": bool(p.get("subbedIn")),
            "subbed_out": bool(p.get("subbedOut")),
            "goals": int(round(st_map.get("totalGoals", 0.0))),
            "assists": int(round(st_map.get("goalAssists", 0.0))),
            "yellow_cards": int(round(st_map.get("yellowCards", 0.0))),
            "red_cards": int(round(st_map.get("redCards", 0.0))),
            "saves": int(round(st_map.get("saves", 0.0))),
            "shots": int(round(st_map.get("totalShots", 0.0))),
        }
        if entry["starter"]:
            starters.append(entry)
        else:
            bench.append(entry)

    return {
        "homeAway": "home" if is_home else "away",
        "team_id": str(t_obj.get("id") or ""),
        "team_name": clean_club_name(t_obj.get("displayName", "")) or ("Home" if is_home else "Away"),
        "formation": str((r_obj or {}).get("formation") or ""),
        "kit_color": kit_color,
        "alt_color": alt_color,
        "starters": starters,
        "bench": bench,
    }


def _parse_espn_summary_payload(data: dict, fallback_match: dict | None = None) -> dict:
    hd = data.get("header") or {}
    comp0 = ((hd.get("competitions") or [{}])[0]) or {}
    st_obj = comp0.get("status") or {}
    st_type = st_obj.get("type") or {}
    st_name = str(st_type.get("name") or "")
    state = str(st_type.get("state") or (fallback_match or {}).get("state") or "pre").lower()
    if state not in ("in", "pre", "post"):
        state = "post" if ("FINAL" in st_name or "FULL_TIME" in st_name) else "pre"

    raw_detail = str(st_type.get("shortDetail") or st_type.get("detail") or (fallback_match or {}).get("status_detail") or "").strip()
    if state == "in":
        status_detail = raw_detail or "LIVE"
    elif state == "post":
        status_detail = raw_detail or "FT"
    else:
        status_detail = raw_detail or "Theo lịch"

    kickoff_utc = str(comp0.get("date") or (fallback_match or {}).get("kickoff_utc") or "")
    dt_utc = _parse_kickoff_dt(kickoff_utc)
    kickoff_vn = dt_utc.astimezone(VN_TZ).strftime("%H:%M · %d/%m/%Y") if dt_utc is not None else ""

    comps = comp0.get("competitors") or []
    h_comp = next((c for c in comps if isinstance(c, dict) and c.get("homeAway") == "home"), None)
    a_comp = next((c for c in comps if isinstance(c, dict) and c.get("homeAway") == "away"), None)
    home = _extract_summary_team_info(h_comp or {}, True)
    away = _extract_summary_team_info(a_comp or {}, False)

    if fallback_match:
        if not home["logo"]:
            home["logo"] = str(fallback_match.get("home_logo") or "")
        if not away["logo"]:
            away["logo"] = str(fallback_match.get("away_logo") or "")
        if not home["name"] or home["name"] == "Home":
            home["name"] = str(fallback_match.get("home_team") or "Home")
        if not away["name"] or away["name"] == "Away":
            away["name"] = str(fallback_match.get("away_team") or "Away")
    home["logo"] = get_team_logo_url(home["name"], fallback_url=home["logo"])
    away["logo"] = get_team_logo_url(away["name"], fallback_url=away["logo"])

    if state in ("in", "post"):
        for tgt, src_c, fb_key in ((home, h_comp, "home_score"), (away, a_comp, "away_score")):
            sc = (src_c or {}).get("score")
            try:
                tgt["score"] = int(float(sc))
            except (TypeError, ValueError):
                tgt["score"] = (fallback_match or {}).get(fb_key)

    details = []
    for d_item in (comp0.get("details") or []):
        if not isinstance(d_item, dict):
            continue
        parts = d_item.get("participants") or []
        p_main = ((parts[0] or {}).get("athlete") or {}) if len(parts) >= 1 and isinstance(parts[0], dict) else {}
        p_ast = ((parts[1] or {}).get("athlete") or {}) if len(parts) >= 2 and isinstance(parts[1], dict) else {}
        t_obj = d_item.get("team") or {}
        details.append({
            "clock": str((d_item.get("clock") or {}).get("displayValue") or ""),
            "team_id": str(t_obj.get("id") or ""),
            "team_name": clean_club_name(t_obj.get("displayName", "")),
            "player": str(p_main.get("shortName") or p_main.get("displayName") or ""),
            "assist": str(p_ast.get("shortName") or p_ast.get("displayName") or ""),
            "scoring_play": bool(d_item.get("scoringPlay")),
            "red_card": bool(d_item.get("redCard")),
            "yellow_card": bool(d_item.get("yellowCard")),
            "penalty": bool(d_item.get("penaltyKick")),
            "own_goal": bool(d_item.get("ownGoal")),
        })

    stats = _parse_boxscore_stats((data.get("boxscore") or {}).get("teams") or [], home["id"], away["id"])

    raw_rosters = data.get("rosters") or []
    h_ros_raw, a_ros_raw = None, None
    for rr in raw_rosters:
        if not isinstance(rr, dict):
            continue
        ha = str(rr.get("homeAway") or "").lower()
        tid = str((rr.get("team") or {}).get("id") or "")
        if ha == "home" or (home["id"] and tid == home["id"]):
            h_ros_raw = rr
        elif ha == "away" or (away["id"] and tid == away["id"]):
            a_ros_raw = rr
    if (h_ros_raw is None or a_ros_raw is None) and len(raw_rosters) >= 2:
        h_ros_raw, a_ros_raw = raw_rosters[0], raw_rosters[1]
    rosters = []
    if h_ros_raw is not None:
        rosters.append(_parse_espn_roster(h_ros_raw, True))
    if a_ros_raw is not None:
        rosters.append(_parse_espn_roster(a_ros_raw, False))

    key_events = []
    for k in (data.get("keyEvents") or []):
        if not isinstance(k, dict):
            continue
        tp_txt = str((k.get("type") or {}).get("text") or "")
        txt = str(k.get("text") or "").strip()
        if not txt and tp_txt in ("Start Delay", "End Delay", "Kickoff"):
            continue
        tp_low = tp_txt.lower()
        key_events.append({
            "clock": str((k.get("clock") or {}).get("displayValue") or ""),
            "type_text": tp_txt,
            "team_name": clean_club_name((k.get("team") or {}).get("displayName", "")),
            "text": txt or tp_txt,
            "scoring_play": bool(k.get("scoringPlay") or "goal" in tp_low),
            "is_red": "red card" in tp_low,
            "is_yellow": "yellow card" in tp_low,
            "is_sub": "substitution" in tp_low,
        })

    commentary = []
    for c in (data.get("commentary") or []):
        if not isinstance(c, dict):
            continue
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        commentary.append({
            "clock": str((c.get("time") or {}).get("displayValue") or ""),
            "text": txt,
            "type_text": str((c.get("type") or {}).get("text") or ""),
        })

    last_five = []
    for lf_item in (data.get("lastFiveGames") or []):
        if not isinstance(lf_item, dict):
            continue
        t_name = clean_club_name((lf_item.get("team") or {}).get("displayName", ""))
        evs = []
        for ge in (lf_item.get("events") or [])[:5]:
            if not isinstance(ge, dict):
                continue
            opp = clean_club_name((ge.get("opponent") or {}).get("displayName", ""))
            evs.append({
                "opponent": opp,
                "score": str(ge.get("score") or ""),
                "result": str(ge.get("gameResult") or "").upper(),
                "at_vs": str(ge.get("atVs") or ""),
                "date": str(ge.get("gameDate") or "")[:10],
            })
        last_five.append({"team_name": t_name, "events": evs})

    gi = data.get("gameInfo") or {}
    venue_full = str((gi.get("venue") or {}).get("fullName") or (fallback_match or {}).get("venue") or "")
    city = str(((gi.get("venue") or {}).get("address") or {}).get("city") or "")
    venue_disp = f"{venue_full} ({city})" if venue_full and city and city not in venue_full else venue_full
    officials = [str(o.get("fullName") or o.get("displayName") or "") for o in (gi.get("officials") or []) if isinstance(o, dict)]
    officials = [o for o in officials if o]
    bcasts = []
    for b in (data.get("broadcasts") or []):
        if isinstance(b, dict):
            m_obj = b.get("media") or {}
            bn = str(m_obj.get("shortName") or m_obj.get("name") or "")
            if bn and bn not in bcasts:
                bcasts.append(bn)

    videos = []
    for v in (data.get("videos") or []):
        if not isinstance(v, dict):
            continue
        lnk = v.get("links") or {}
        v_url = str((((lnk.get("source") or {}).get("mezzanine")) or {}).get("href") or (lnk.get("web") or {}).get("href") or "")
        if v_url:
            videos.append({
                "headline": str(v.get("headline") or "Highlight"),
                "description": str(v.get("description") or ""),
                "duration": v.get("duration"),
                "thumbnail": str(v.get("thumbnail") or ""),
                "url": v_url,
            })

    now_vn = datetime.now(VN_TZ).strftime("%H:%M:%S %d/%m")
    return {
        "fetched_vn": now_vn,
        "fetched_mono": time.monotonic(),
        "state": state,
        "status_detail": status_detail,
        "status_desc": str(st_type.get("description") or ""),
        "kickoff_vn": kickoff_vn,
        "home": home,
        "away": away,
        "details": details,
        "stats": stats,
        "rosters": rosters,
        "key_events": key_events,
        "commentary": commentary,
        "last_five": last_five,
        "game_info": {
            "venue": venue_disp,
            "attendance": gi.get("attendance"),
            "officials": officials,
            "broadcasts": bcasts,
        },
        "videos": videos,
        "error": "",
    }


def get_espn_match_summary(
    slug: str,
    event_id: str,
    force_refresh: bool = False,
    timeout: float = 3.5,
    fallback_match: dict | None = None,
) -> dict:
    """Fetch & cache ESPN live match summary safely across concurrent viewers."""
    slug_s = str(slug or "").strip()
    eid_s = str(event_id or "").strip()
    if not slug_s or not eid_s:
        return {"error": "Thiếu mã giải hoặc mã trận ESPN để tải trực tiếp."}

    key = (slug_s, eid_s)
    now_m = time.monotonic()

    with _SUMMARY_LOCK:
        cached = dict(_SUMMARY_CACHE[key]) if key in _SUMMARY_CACHE else None
        last_force = _SUMMARY_FORCE_MONO.get(key, 0.0)

    if cached is not None:
        ttl = _SUMMARY_LIVE_TTL_SEC if cached.get("state") == "in" else _SUMMARY_STATIC_TTL_SEC
        age = now_m - float(cached.get("fetched_mono", 0.0))
        if force_refresh:
            if (now_m - last_force) < _SUMMARY_FORCE_COOLDOWN_SEC:
                cached["cooldown_wait"] = int(_SUMMARY_FORCE_COOLDOWN_SEC - (now_m - last_force))
                return cached
        elif age <= ttl:
            return cached

    if force_refresh:
        with _SUMMARY_LOCK:
            _SUMMARY_FORCE_MONO[key] = now_m

    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug_s}/summary?event={eid_s}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_json = json.loads(resp.read().decode("utf-8"))
        parsed = _parse_espn_summary_payload(raw_json, fallback_match=fallback_match)
        with _SUMMARY_LOCK:
            _SUMMARY_CACHE[key] = parsed
        return dict(parsed)
    except Exception as exc:
        if cached is not None:
            cached["error"] = f"Không thể đồng bộ mới từ ESPN ({type(exc).__name__}), đang giữ dữ liệu gần nhất."
            return cached
        return {"error": f"Chưa kết nối được tới ESPN Live Summary ({type(exc).__name__})."}


def _catalog_played_lookup(catalog: dict) -> dict:
    exact_idx = {}
    pair_idx = {}
    for comp_name, mlist in (catalog or {}).items():
        for m in mlist:
            if m.get("is_upcoming"):
                continue
            hk = _team_alias_key(m.get("home_team", ""))
            ak = _team_alias_key(m.get("away_team", ""))
            d_s = str(m.get("date") or "")[:10]
            if not hk or not ak:
                continue
            val = (comp_name, m.get("label", ""), d_s)
            if d_s:
                exact_idx[(hk, ak, d_s)] = val
            prev = pair_idx.get((hk, ak))
            if prev is None or d_s > prev[2]:
                pair_idx[(hk, ak)] = val
    return {"exact": exact_idx, "pair": pair_idx}


def _find_catalog_ref(cat_idx: dict, hk: str, ak: str, d_s: str):
    hit = cat_idx["exact"].get((hk, ak, d_s))
    if hit:
        return (hit[0], hit[1])
    hit2 = cat_idx["pair"].get((hk, ak))
    if hit2 and d_s and hit2[2]:
        try:
            diff = abs((date.fromisoformat(d_s) - date.fromisoformat(hit2[2])).days)
            if diff <= 1:
                return (hit2[0], hit2[1])
        except Exception:
            pass
    return None


def _parse_iso_date(d_s: str):
    if not d_s or len(d_s) < 10:
        return None
    try:
        return date.fromisoformat(d_s[:10])
    except Exception:
        return None


def build_merged_daily_fixtures(
    live_data: dict,
    up_df: pd.DataFrame,
    catalog: dict,
    hist_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Merge live ESPN fixtures, ML upcoming rows, and 2026/27 recent historical results without duplicates."""
    cat_idx = _catalog_played_lookup(catalog)

    # Index upcoming_fixtures.parquet by (_team_alias_key(home), _team_alias_key(away))
    up_by_pair: dict[tuple[str, str], list[dict]] = {}
    up_records: list[dict] = []
    if up_df is not None and not up_df.empty:
        sub_cols = [c for c in ("match_id", "match_date", "league", "season", "home_team", "away_team", "pred_home_goals", "pred_away_goals", "prob_H", "prob_D", "prob_A", "home_elo", "away_elo") if c in up_df.columns]
        for rr in up_df[sub_cols].to_dict(orient="records"):
            hk = _team_alias_key(rr.get("home_team", ""))
            ak = _team_alias_key(rr.get("away_team", ""))
            d_s = str(rr.get("match_date") or "")[:10]
            rr["_hk"] = hk
            rr["_ak"] = ak
            rr["_d"] = d_s
            up_records.append(rr)
            up_by_pair.setdefault((hk, ak), []).append(rr)

    merged: list[dict] = []
    matched_up_ids: set[str] = set()
    seen_dates_by_pair: dict[tuple[str, str], list[date]] = {}

    def _mark_seen(hk: str, ak: str, *d_strs: str):
        for ds in d_strs:
            dt_o = _parse_iso_date(ds)
            if dt_o is not None:
                seen_dates_by_pair.setdefault((hk, ak), []).append(dt_o)

    for lf in (live_data.get("live_fixtures") or []):
        entry = _normalize_live_entry(lf)
        hk = _team_alias_key(entry.get("home_team", ""))
        ak = _team_alias_key(entry.get("away_team", ""))
        vn_d = str(entry.get("vn_date") or "")[:10]
        utc_d = str(entry.get("kickoff_utc") or "")[:10]

        # Match upcoming ML record within +-1 day (handles midnight UTC -> GMT+7 rollover)
        up_match = None
        for cand in up_by_pair.get((hk, ak), []):
            cd = cand["_d"]
            if not cd:
                continue
            try:
                base_d = vn_d or utc_d
                diff1 = abs((date.fromisoformat(base_d) - date.fromisoformat(cd)).days) if base_d else 99
                diff2 = abs((date.fromisoformat(utc_d) - date.fromisoformat(cd)).days) if utc_d else 99
                if min(diff1, diff2) <= 1:
                    up_match = cand
                    break
            except Exception:
                continue

        if up_match is not None:
            uid = str(up_match["match_id"])
            matched_up_ids.add(uid)
            elo_h = _safe_float(up_match.get("home_elo"), 1650.0)
            elo_a = _safe_float(up_match.get("away_elo"), 1650.0)
            entry["upcoming_match_id"] = uid
            entry["prob_H"] = round(_safe_float(up_match.get("prob_H"), 0.35) * 100.0, 1)
            entry["prob_D"] = round(_safe_float(up_match.get("prob_D"), 0.28) * 100.0, 1)
            entry["prob_A"] = round(_safe_float(up_match.get("prob_A"), 0.37) * 100.0, 1)
            entry["pred_home_goals"] = round(_safe_float(up_match.get("pred_home_goals"), 1.35), 2)
            entry["pred_away_goals"] = round(_safe_float(up_match.get("pred_away_goals"), 1.15), 2)
            entry["comb_elo"] = round((elo_h + elo_a) / 2.0, 1)
            _mark_seen(hk, ak, vn_d, utc_d, up_match["_d"])
        else:
            entry["upcoming_match_id"] = ""
            entry["prob_H"] = None
            entry["prob_D"] = None
            entry["prob_A"] = None
            entry["pred_home_goals"] = None
            entry["pred_away_goals"] = None
            entry["comb_elo"] = 1720.0 if entry.get("league_code") in ("ENG", "ESP", "UCL", "ITA", "GER", "FRA") else 1600.0
            _mark_seen(hk, ak, vn_d, utc_d)

        entry["catalog_ref"] = _find_catalog_ref(cat_idx, hk, ak, vn_d or utc_d)
        merged.append(entry)

    # Append remaining upcoming fixtures (other leagues & future dates beyond ESPN 7-day window)
    for rr in up_records:
        uid = str(rr["match_id"])
        if uid in matched_up_ids:
            continue
        lg = str(rr.get("league") or "ENG")
        d_s = rr["_d"]
        _mark_seen(rr["_hk"], rr["_ak"], d_s)
        comp_lbl = CATALOG_CATEGORY_MAP.get(lg, LEAGUE_META.get(lg, (lg, lg))[0])
        if lg == "INTL_FR":
            comp_lbl = "🌍 Giao Hữu Quốc Tế (International)"
        elo_h = _safe_float(rr.get("home_elo"), 1650.0)
        elo_a = _safe_float(rr.get("away_elo"), 1650.0)
        cat_ref = _find_catalog_ref(cat_idx, rr["_hk"], rr["_ak"], d_s)
        merged.append({
            "espn_event_id": "",
            "league_code": lg,
            "espn_slug": LEAGUE_META.get(lg, ("", "", ""))[2] if lg in LEAGUE_META else "",
            "competition": comp_lbl,
            "kickoff_utc": "",
            "kickoff_ts": 0.0,
            "match_date": d_s,
            "vn_date": d_s,
            "vn_time": "",
            "state": "post" if cat_ref else "pre",
            "status": "STATUS_FINAL" if cat_ref else "STATUS_SCHEDULED",
            "status_detail": "FT" if cat_ref else "Theo lịch",
            "home_team": str(rr.get("home_team") or ""),
            "away_team": str(rr.get("away_team") or ""),
            "home_logo": get_team_logo_url(rr.get("home_team") or ""),
            "away_logo": get_team_logo_url(rr.get("away_team") or ""),
            "home_score": None,
            "away_score": None,
            "home_form": "",
            "away_form": "",
            "venue": "",
            "odds": {},
            "upcoming_match_id": uid,
            "prob_H": round(_safe_float(rr.get("prob_H"), 0.35) * 100.0, 1),
            "prob_D": round(_safe_float(rr.get("prob_D"), 0.28) * 100.0, 1),
            "prob_A": round(_safe_float(rr.get("prob_A"), 0.37) * 100.0, 1),
            "pred_home_goals": round(_safe_float(rr.get("pred_home_goals"), 1.35), 2),
            "pred_away_goals": round(_safe_float(rr.get("pred_away_goals"), 1.15), 2),
            "comb_elo": round((elo_h + elo_a) / 2.0, 1),
            "catalog_ref": cat_ref,
        })

    # Append recent 2026/27 played fixtures for past-week slider browsing without duplicates
    if hist_df is not None and not hist_df.empty and "match_date" in hist_df.columns:
        sub_h = hist_df[hist_df["match_date"] >= "2026-08-01"]
        h_cols = [c for c in ("match_id", "match_date", "league", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg", "home_elo", "away_elo", "_hk", "_ak") if c in sub_h.columns]
        for rr in sub_h[h_cols].to_dict(orient="records"):
            d_s = str(rr.get("match_date") or "")[:10]
            dt_h = _parse_iso_date(d_s)
            if dt_h is None:
                continue
            hk = rr.get("_hk") or _team_alias_key(rr.get("home_team", ""))
            ak = rr.get("_ak") or _team_alias_key(rr.get("away_team", ""))
            seen_list = seen_dates_by_pair.get((hk, ak), [])
            if any(abs((dt_h - sd).days) <= 1 for sd in seen_list):
                continue
            seen_dates_by_pair.setdefault((hk, ak), []).append(dt_h)

            lg = str(rr.get("league") or "ENG")
            comp_lbl = CATALOG_CATEGORY_MAP.get(lg, LEAGUE_META.get(lg, (lg, lg))[0])
            if lg == "INTL_FR":
                comp_lbl = "🌍 Giao Hữu Quốc Tế (International)"
            elo_h = _safe_float(rr.get("home_elo"), 1650.0)
            elo_a = _safe_float(rr.get("away_elo"), 1650.0)
            h_xg = _safe_float(rr.get("home_xg"), -1.0)
            a_xg = _safe_float(rr.get("away_xg"), -1.0)
            cat_ref = _find_catalog_ref(cat_idx, hk, ak, d_s)
            merged.append({
                "espn_event_id": "",
                "league_code": lg,
                "espn_slug": LEAGUE_META.get(lg, ("", "", ""))[2] if lg in LEAGUE_META else "",
                "competition": comp_lbl,
                "kickoff_utc": "",
                "kickoff_ts": 0.0,
                "match_date": d_s,
                "vn_date": d_s,
                "vn_time": "",
                "state": "post",
                "status": "STATUS_FINAL",
                "status_detail": "FT",
                "home_team": str(rr.get("home_team") or ""),
                "away_team": str(rr.get("away_team") or ""),
                "home_logo": get_team_logo_url(rr.get("home_team") or ""),
                "away_logo": get_team_logo_url(rr.get("away_team") or ""),
                "home_score": int(round(_safe_float(rr.get("home_goals"), 0))),
                "away_score": int(round(_safe_float(rr.get("away_goals"), 0))),
                "home_form": "",
                "away_form": "",
                "venue": "",
                "odds": {},
                "upcoming_match_id": str(rr.get("match_id") or ""),
                "prob_H": None,
                "prob_D": None,
                "prob_A": None,
                "pred_home_goals": None,
                "pred_away_goals": None,
                "hist_home_xg": round(h_xg, 2) if h_xg >= 0 else None,
                "hist_away_xg": round(a_xg, 2) if a_xg >= 0 else None,
                "comb_elo": round((elo_h + elo_a) / 2.0, 1),
                "catalog_ref": cat_ref,
            })

    return merged


def _sort_match_key(m: dict):
    st_rank = {"in": 0, "pre": 1, "post": 2}.get(m.get("state", "pre"), 1)
    ts = float(m.get("kickoff_ts") or 0.0)
    if ts <= 0:
        d_s = str(m.get("vn_date") or "9999-12-31")
        t_s = str(m.get("vn_time") or "23:59")
        try:
            ts = datetime.strptime(f"{d_s} {t_s}", "%Y-%m-%d %H:%M").replace(tzinfo=VN_TZ).timestamp()
        except Exception:
            ts = 9e12
    return (st_rank, ts, -float(m.get("comb_elo") or 1600.0), str(m.get("home_team") or ""))


def _format_relative_kickoff(m: dict, now_ts: float) -> str:
    state = m.get("state", "pre")
    if state == "in":
        return f"🔴 LIVE · {m.get('status_detail') or 'Đang đá'}"
    if state == "post":
        return f"✅ {m.get('status_detail') or 'Hết giờ'}"
    ts = float(m.get("kickoff_ts") or 0.0)
    if ts > 0:
        diff = ts - now_ts
        if -7200 <= diff <= 0:
            return "🟡 Sắp bóng lăn"
        if 0 < diff < 3600:
            return f"⏳ Còn {max(1, int(diff // 60))}p"
        if diff < 86400:
            return f"⏳ Còn {int(diff // 3600)}g {int((diff % 3600) // 60)}p"
        if diff >= 86400:
            return f"📅 Còn {int(diff // 86400)} ngày"
    return "📅 Đã lên lịch"


def _team_badge_html(name: str, logo_url: str, is_home: bool, size: int = 24) -> str:
    parts = [w for w in str(name or "").split() if w]
    ini = (parts[0][:2] if len(parts) == 1 else (parts[0][0] + parts[-1][0])).upper() if parts else "FC"
    bg = "#13293D" if is_home else "#331827"
    bd = "#38BDF8" if is_home else "#F43F5E"
    fg = "#7DD3FC" if is_home else "#FDA4AF"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="14.5" fill="{bg}" stroke="{bd}" stroke-width="1.8"/>'
        f'<text x="16" y="20" text-anchor="middle" fill="{fg}" font-family="Inter,Arial,sans-serif" font-weight="800" font-size="11">{html.escape(ini)}</text>'
        f'</svg>'
    )
    fallback_uri = f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('ascii')}"
    resolved_logo = get_team_logo_url(name, fallback_url=logo_url)
    if resolved_logo:
        return (
            f'<img src="{html.escape(resolved_logo, quote=True)}" width="{size}" height="{size}" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;border-radius:4px;flex:none;" '
            f'alt="{html.escape(str(name or ""))}" onerror="this.onerror=null;this.src=\'{fallback_uri}\';">'
        )
    return (
        f'<img src="{fallback_uri}" width="{size}" height="{size}" '
        f'style="width:{size}px;height:{size}px;object-fit:contain;flex:none;" alt="{html.escape(str(name or ""))}">'
    )


def _render_realtime_clock_banner(live_cnt: int, today_pre_cnt: int, today_post_cnt: int, updated_vn: str):
    upd_esc = html.escape(str(updated_vn or "Vừa xong"))
    now_vn_srv = datetime.now(VN_TZ)
    init_time = now_vn_srv.strftime("%H:%M:%S")
    init_date = f"{_VI_WEEKDAYS[now_vn_srv.weekday()]}, {now_vn_srv.strftime('%d/%m/%Y')} · GMT+7"
    live_cls = "nckh-stat-chip live" if live_cnt > 0 else "nckh-stat-chip"
    banner_html = compact_html(f"""
<style>
.nckh-clk-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px 14px;
    padding: 10px 16px;
    margin-bottom: 10px;
    border-radius: 12px;
    background: radial-gradient(circle at 12% -20%, rgba(56,189,248,0.13), #0F1624 68%);
    border: 1px solid rgba(56,189,248,0.24);
}}
.nckh-clk-left {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.nckh-live-pill {{
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: 999px;
    background: rgba(16,185,129,0.14); border: 1px solid rgba(16,185,129,0.38);
    color: #34D399; font-size: 10px; font-weight: 800; letter-spacing: 0.08em;
}}
.nckh-pulse-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: #10B981;
    animation: nckhPGlow 1.6s infinite;
}}
@keyframes nckhPGlow {{
    0% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }}
    70% {{ box-shadow: 0 0 0 7px rgba(16,185,129,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
}}
#nckh-vn-clock {{ font-size: 18px; font-weight: 800; color: #FFFFFF; font-variant-numeric: tabular-nums; letter-spacing: 0.02em; }}
#nckh-vn-date {{ font-size: 12.5px; font-weight: 600; color: #94A3B8; }}
.nckh-clk-right {{ display: flex; align-items: center; gap: 7px; flex-wrap: wrap; font-size: 11.5px; color: #CBD5E1; font-weight: 600; }}
.nckh-stat-chip {{ background: rgba(17,24,39,0.85); border: 1px solid rgba(148,163,184,0.16); padding: 3px 9px; border-radius: 999px; white-space: nowrap; }}
.nckh-stat-chip.live {{ background: rgba(239,68,68,0.16); border-color: rgba(239,68,68,0.48); color: #FCA5A5; font-weight: 800; }}
.nckh-upd-tag {{ color: #64748B; font-size: 11px; white-space: nowrap; }}
@media (max-width: 768px) {{
    .nckh-clk-bar {{ padding: 8px 11px; gap: 7px; }}
    #nckh-vn-clock {{ font-size: 15.5px; }}
    #nckh-vn-date {{ font-size: 11px; }}
}}
</style>
<div class="nckh-clk-bar">
  <div class="nckh-clk-left">
    <span class="nckh-live-pill"><span class="nckh-pulse-dot"></span>THỜI GIAN THỰC</span>
    <span id="nckh-vn-clock">{init_time}</span>
    <span id="nckh-vn-date">{html.escape(init_date)}</span>
  </div>
  <div class="nckh-clk-right">
    <span class="{live_cls}">🔴 {live_cnt} đang đá</span>
    <span class="nckh-stat-chip">⏰ {today_pre_cnt} sắp đá hôm nay</span>
    <span class="nckh-stat-chip">✅ {today_post_cnt} đã đá</span>
    <span class="nckh-upd-tag">Đồng bộ: {upd_esc}</span>
  </div>
</div>
<script>
(function() {{
  if (window.__nckhClockTimer) clearInterval(window.__nckhClockTimer);
  var fmtTime = new Intl.DateTimeFormat('vi-VN', {{
    timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  }});
  var fmtDate = new Intl.DateTimeFormat('vi-VN', {{
    timeZone: 'Asia/Ho_Chi_Minh', weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric'
  }});
  function tick() {{
    var elT = document.getElementById('nckh-vn-clock');
    var elD = document.getElementById('nckh-vn-date');
    if (!elT || !elD) return;
    var now = new Date();
    elT.textContent = fmtTime.format(now);
    elD.textContent = fmtDate.format(now) + ' · GMT+7';
  }}
  tick();
  window.__nckhClockTimer = setInterval(tick, 1000);
  if (!window.__nckhCardClickBound) {{
    window.__nckhCardClickBound = true;
    function activateMatchCard(card) {{
      if (!card) return;
      var row = card.closest('[data-testid="stHorizontalBlock"]');
      if (!row) return;
      var btn = row.querySelector('button[data-testid="stBaseButton-primary"], button[kind="primary"], button[data-testid^="baseButton-primary"]')
                || row.querySelector('button');
      if (btn) {{
        card.classList.add('dm-pressing');
        setTimeout(function() {{ card.classList.remove('dm-pressing'); }}, 220);
        btn.click();
      }}
    }}
    document.addEventListener('click', function(e) {{
      var card = e.target && e.target.closest ? e.target.closest('.dm-card.dm-clickable') : null;
      if (card) activateMatchCard(card);
    }});
    document.addEventListener('keydown', function(e) {{
      if ((e.key === 'Enter' || e.key === ' ') && e.target && e.target.classList && e.target.classList.contains('dm-clickable')) {{
        e.preventDefault();
        activateMatchCard(e.target);
      }}
    }});
  }}
}})();
</script>
""")
    st.html(banner_html, unsafe_allow_javascript=True)


def _open_inline_live_viewer(m: dict):
    st.session_state["daily_live_viewer_event"] = {
        "event_id": str(m.get("espn_event_id") or ""),
        "slug": str(m.get("espn_slug") or ""),
        "match": m,
    }
    st.session_state["daily_live_viewer_force"] = False


def _close_inline_live_viewer():
    st.session_state.pop("daily_live_viewer_event", None)
    st.session_state.pop("daily_live_viewer_force", None)


def _kit_text_color(hex_col: str) -> str:
    h = str(hex_col or "").lstrip("#")
    if len(h) != 6:
        return "#FFFFFF"
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return "#0F172A" if lum > 0.62 else "#FFFFFF"
    except Exception:
        return "#FFFFFF"


def _espn_pos_coords(pos_abbr: str) -> tuple[float, float]:
    p = str(pos_abbr or "").strip().upper()
    coord_map = {
        "G": (0.068, 0.50), "GK": (0.068, 0.50),
        "LB": (0.175, 0.16), "LWB": (0.215, 0.14),
        "CD-L": (0.165, 0.37), "LCB": (0.165, 0.37), "D-L": (0.165, 0.37),
        "CD": (0.160, 0.50), "CB": (0.160, 0.50), "D": (0.160, 0.50),
        "CD-R": (0.165, 0.63), "RCB": (0.165, 0.63), "D-R": (0.165, 0.63),
        "RB": (0.175, 0.84), "RWB": (0.215, 0.86),
        "DM-L": (0.255, 0.35), "LDM": (0.255, 0.35),
        "DM": (0.250, 0.50), "CDM": (0.250, 0.50),
        "DM-R": (0.255, 0.65), "RDM": (0.255, 0.65),
        "LM": (0.310, 0.17), "LCM": (0.305, 0.37), "CM-L": (0.305, 0.37),
        "CM": (0.300, 0.50), "M": (0.300, 0.50),
        "CM-R": (0.305, 0.63), "RCM": (0.305, 0.63), "RM": (0.310, 0.83),
        "AM-L": (0.375, 0.24), "LAM": (0.375, 0.24),
        "AM": (0.375, 0.50), "CAM": (0.375, 0.50),
        "AM-R": (0.375, 0.76), "RAM": (0.375, 0.76),
        "LW": (0.405, 0.20), "LF": (0.405, 0.26),
        "F": (0.432, 0.50), "CF": (0.432, 0.50), "ST": (0.432, 0.50),
        "RF": (0.405, 0.74), "RW": (0.405, 0.80),
    }
    if p in coord_map:
        return coord_map[p]
    base_x = 0.068 if p.startswith("G") else (0.170 if "D" in p or "B" in p else (0.420 if any(k in p for k in ("F", "W", "S")) else 0.310))
    base_y = 0.24 if "-L" in p or p.startswith("L") else (0.76 if "-R" in p or p.startswith("R") else 0.50)
    return (base_x, base_y)


def _spread_pitch_coords(starters: list[dict], is_home: bool) -> list[tuple[dict, float, float]]:
    raw = []
    for p in starters[:11]:
        hx, hy = _espn_pos_coords(p.get("pos", ""))
        raw.append([p, hx, hy])
    # Separate overlapping nodes sharing the same depth/lane (e.g., two CDs, two CMs, or two Fs)
    raw.sort(key=lambda item: (round(item[1], 2), item[2]))
    used = [False] * len(raw)
    for i in range(len(raw)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(raw)):
            if not used[j] and abs(raw[j][1] - raw[i][1]) < 0.045 and abs(raw[j][2] - raw[i][2]) < 0.11:
                cluster.append(j)
        if len(cluster) > 1:
            avg_y = sum(raw[k][2] for k in cluster) / len(cluster)
            span = 0.155
            start_y = avg_y - span * (len(cluster) - 1) / 2.0
            for idx_c, k in enumerate(cluster):
                raw[k][2] = max(0.11, min(0.89, start_y + idx_c * span))
                used[k] = True
    out = []
    for p, hx, hy in raw:
        px = hx if is_home else (1.0 - hx)
        py = hy if is_home else (1.0 - hy)
        out.append((p, px, py))
    return out


def _build_live_tactical_pitch_svg(home_roster: dict, away_roster: dict) -> str:
    W, H = 840, 450
    pad_x, pad_y = 28, 24
    pw, ph = W - 2 * pad_x, H - 2 * pad_y

    h_col = str(home_roster.get("kit_color") or "#0284C7")
    a_col = str(away_roster.get("kit_color") or "#E11D48")
    if h_col.lower() == a_col.lower():
        a_col = str(away_roster.get("alt_color") or "#E11D48")
        if h_col.lower() == a_col.lower():
            a_col = "#E11D48"

    h_txt_c = _kit_text_color(h_col)
    a_txt_c = _kit_text_color(a_col)

    stripes = []
    n_stripes = 10
    sw = pw / n_stripes
    for i in range(n_stripes):
        fill = "#0D2B22" if i % 2 == 0 else "#0B241C"
        stripes.append(f'<rect x="{pad_x + i*sw:.1f}" y="{pad_y}" width="{sw:.1f}" height="{ph}" fill="{fill}"/>')

    nodes = []
    for roster, is_home, fill_c, txt_c, halo_c in (
        (home_roster, True, h_col, h_txt_c, "#38BDF8"),
        (away_roster, False, a_col, a_txt_c, "#FB7185"),
    ):
        placed = _spread_pitch_coords(roster.get("starters") or [], is_home=is_home)
        for p, nx, ny in placed:
            cx = pad_x + nx * pw
            cy = pad_y + ny * ph
            jersey = html.escape(str(p.get("jersey") or "•"))
            sname = str(p.get("short_name") or p.get("name") or "").strip()
            if len(sname) > 13:
                sname = sname.split()[-1][:12]
            sname_esc = html.escape(sname)

            badges = []
            if p.get("goals", 0) > 0:
                badges.append(f'<circle cx="{cx+12:.1f}" cy="{cy-12:.1f}" r="6.5" fill="#10B981" stroke="#0B1220" stroke-width="1.5"/>'
                              f'<text x="{cx+12:.1f}" y="{cy-9:.1f}" text-anchor="middle" font-size="8.5" font-weight="800" fill="#052E16">G</text>')
            if p.get("assists", 0) > 0:
                badges.append(f'<circle cx="{cx+12:.1f}" cy="{cy+2:.1f}" r="5.5" fill="#38BDF8" stroke="#0B1220" stroke-width="1.2"/>'
                              f'<text x="{cx+12:.1f}" y="{cy+5:.1f}" text-anchor="middle" font-size="7.5" font-weight="800" fill="#082F49">A</text>')
            if p.get("red_cards", 0) > 0:
                badges.append(f'<rect x="{cx-16:.1f}" y="{cy-17:.1f}" width="7" height="10" rx="1.5" fill="#EF4444" stroke="#0B1220" stroke-width="1"/>')
            elif p.get("yellow_cards", 0) > 0:
                badges.append(f'<rect x="{cx-16:.1f}" y="{cy-17:.1f}" width="7" height="10" rx="1.5" fill="#FACC15" stroke="#0B1220" stroke-width="1"/>')
            if p.get("subbed_out"):
                badges.append(f'<circle cx="{cx-12:.1f}" cy="{cy+11:.1f}" r="5" fill="#F97316" stroke="#0B1220" stroke-width="1"/>'
                              f'<text x="{cx-12:.1f}" y="{cy+13.8:.1f}" text-anchor="middle" font-size="7" font-weight="900" fill="#FFFFFF">↓</text>')

            lbl_w = max(54, len(sname) * 6.4 + 12)
            nodes.append(
                f'<g>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="16.5" fill="{fill_c}" stroke="{halo_c}" stroke-width="2.4"/>'
                f'<text x="{cx:.1f}" y="{cy+4.5:.1f}" text-anchor="middle" font-family="Plus Jakarta Sans,sans-serif" font-size="12" font-weight="800" fill="{txt_c}">{jersey}</text>'
                f'<rect x="{cx - lbl_w/2:.1f}" y="{cy+18.5:.1f}" width="{lbl_w:.1f}" height="15" rx="4" fill="rgba(9,13,22,0.86)" stroke="rgba(148,163,184,0.28)" stroke-width="0.8"/>'
                f'<text x="{cx:.1f}" y="{cy+28.8:.1f}" text-anchor="middle" font-family="Plus Jakarta Sans,sans-serif" font-size="9.5" font-weight="700" fill="#F1F5F9">{sname_esc}</text>'
                f'{"".join(badges)}'
                f'</g>'
            )

    h_hdr = html.escape(f"{home_roster.get('team_name', 'Chủ nhà')} ({home_roster.get('formation') or 'XI'})")
    a_hdr = html.escape(f"{away_roster.get('team_name', 'Khách')} ({away_roster.get('formation') or 'XI'})")
    mid_x = W / 2
    mid_y = H / 2
    line_col = "rgba(255,255,255,0.28)"

    return compact_html(f"""
    <svg viewBox="0 0 {W} {H}" style="width:100%;max-width:860px;height:auto;display:block;margin:6px auto;border-radius:12px;border:1px solid rgba(148,163,184,0.24);background:#091E18;" xmlns="http://www.w3.org/2000/svg">
        {"".join(stripes)}
        <rect x="{pad_x}" y="{pad_y}" width="{pw}" height="{ph}" fill="none" stroke="{line_col}" stroke-width="2"/>
        <line x1="{mid_x}" y1="{pad_y}" x2="{mid_x}" y2="{pad_y+ph}" stroke="{line_col}" stroke-width="2"/>
        <circle cx="{mid_x}" cy="{mid_y}" r="54" fill="none" stroke="{line_col}" stroke-width="2"/>
        <circle cx="{mid_x}" cy="{mid_y}" r="3.5" fill="{line_col}"/>
        <rect x="{pad_x}" y="{mid_y-115}" width="112" height="230" fill="none" stroke="{line_col}" stroke-width="1.8"/>
        <rect x="{pad_x}" y="{mid_y-52}" width="38" height="104" fill="none" stroke="{line_col}" stroke-width="1.8"/>
        <rect x="{pad_x+pw-112}" y="{mid_y-115}" width="112" height="230" fill="none" stroke="{line_col}" stroke-width="1.8"/>
        <rect x="{pad_x+pw-38}" y="{mid_y-52}" width="38" height="104" fill="none" stroke="{line_col}" stroke-width="1.8"/>
        <text x="{pad_x+10}" y="{pad_y+16}" font-family="Plus Jakarta Sans,sans-serif" font-size="11.5" font-weight="800" fill="#7DD3FC">{h_hdr}</text>
        <text x="{pad_x+pw-10}" y="{pad_y+16}" text-anchor="end" font-family="Plus Jakarta Sans,sans-serif" font-size="11.5" font-weight="800" fill="#FDA4AF">{a_hdr}</text>
        {"".join(nodes)}
    </svg>
    """)


def _render_live_viewer_body(slug: str, event_id: str, fallback_match: dict, force_flag: bool):
    summ = get_espn_match_summary(slug, event_id, force_refresh=force_flag, fallback_match=fallback_match)
    if summ.get("cooldown_wait"):
        st.toast(f"Dữ liệu trận đấu vừa đồng bộ xong (thử lại sau {summ['cooldown_wait']}s).")
    if summ.get("error") and not summ.get("home"):
        st.warning(summ["error"])
        return
    if summ.get("error"):
        st.caption(f"⚠️ {summ['error']}")

    home = summ.get("home") or {}
    away = summ.get("away") or {}
    state = summ.get("state") or fallback_match.get("state") or "in"
    st_det = str(summ.get("status_detail") or fallback_match.get("status_detail") or "LIVE")
    comp_lbl = str(fallback_match.get("competition") or "")
    gi = summ.get("game_info") or {}
    venue_txt = gi.get("venue") or fallback_match.get("venue") or ""
    bcasts = gi.get("broadcasts") or []

    if state == "in":
        st_pill = f'<span class="nckh-live-pill" style="background:rgba(239,68,68,0.18);border-color:rgba(239,68,68,0.5);color:#FCA5A5;"><span class="nckh-pulse-dot" style="background:#EF4444;"></span>TRỰC TIẾP · {html.escape(st_det)}</span>'
    elif state == "post":
        st_pill = f'<span class="nckh-live-pill" style="background:rgba(16,185,129,0.16);border-color:rgba(16,185,129,0.45);color:#34D399;">✅ KẾT THÚC · {html.escape(st_det)}</span>'
    else:
        st_pill = f'<span class="nckh-live-pill" style="background:rgba(56,189,248,0.14);border-color:rgba(56,189,248,0.4);color:#7DD3FC;">⏰ {html.escape(summ.get("kickoff_vn") or st_det)}</span>'

    h_logo = _team_badge_html(home.get("name", ""), home.get("logo", ""), is_home=True, size=38)
    a_logo = _team_badge_html(away.get("name", ""), away.get("logo", ""), is_home=False, size=38)
    hs, as_ = home.get("score"), away.get("score")
    score_html = f'{int(hs)} - {int(as_)}' if (hs is not None and as_ is not None) else "VS"
    score_cls = "#FCA5A5" if state == "in" else ("#34D399" if state == "post" else "#94A3B8")

    # Inline goals & red cards summary
    h_evs, a_evs = [], []
    h_id, a_id = str(home.get("id") or ""), str(away.get("id") or "")
    for d in (summ.get("details") or []):
        if not d.get("scoring_play") and not d.get("red_card"):
            continue
        ico = "⚽" if d.get("scoring_play") else "🟥"
        tag = f" {d.get('clock','')}" + (" (P)" if d.get("penalty") else "") + (" (OG)" if d.get("own_goal") else "")
        line = f'{ico} {html.escape(str(d.get("player") or ""))} <b>{html.escape(tag)}</b>'
        is_h = (h_id and str(d.get("team_id") or "") == h_id) or (not d.get("team_id") and d.get("team_name") == home.get("name"))
        if is_h:
            h_evs.append(line)
        else:
            a_evs.append(line)

    ai_strip = ""
    if fallback_match.get("prob_H") is not None:
        ai_strip = (
            f'<div style="display:flex;justify-content:center;flex-wrap:wrap;gap:12px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(148,163,184,0.12);font-size:11.5px;color:#94A3B8;">'
            f'<span>🔮 AI trước trận: <b style="color:#7DD3FC;">{fallback_match["prob_H"]:.0f}%</b> · <b>{fallback_match["prob_D"]:.0f}%</b> · <b style="color:#FDA4AF;">{fallback_match["prob_A"]:.0f}%</b></span>'
            f'<span>📊 Dự báo xG: <b style="color:#E2E8F0;">{fallback_match["pred_home_goals"]:.2f} - {fallback_match["pred_away_goals"]:.2f}</b></span>'
            f'</div>'
        )

    meta_parts = [p for p in (comp_lbl, venue_txt, f"📺 {', '.join(bcasts[:3])}" if bcasts else "", f"Đồng bộ {summ.get('fetched_vn', '')}") if p]
    meta_html = " &nbsp;·&nbsp; ".join(html.escape(str(x)) for x in meta_parts)
    h_rec = f'<div style="color:#64748B;font-size:11px;font-weight:600;">{html.escape(home.get("record") or "")}</div>' if home.get("record") else ""
    a_rec = f'<div style="color:#64748B;font-size:11px;font-weight:600;">{html.escape(away.get("record") or "")}</div>' if away.get("record") else ""

    ev_row_html = ""
    if h_evs or a_evs:
        ev_row_html = (
            f'<div class="lv-ev-summary">'
            f'<div style="text-align:right;color:#CBD5E1;">{"<br>".join(h_evs)}</div>'
            f'<div style="color:#475569;text-align:center;">•</div>'
            f'<div style="text-align:left;color:#CBD5E1;">{"<br>".join(a_evs)}</div>'
            f'</div>'
        )

    st.markdown(compact_html(f"""
    <div class="lv-hero">
        <div class="lv-hero-top">
            {st_pill}
            <span style="color:#94A3B8;font-size:11.5px;font-weight:600;">{meta_html}</span>
        </div>
        <div class="lv-score-grid">
            <div class="lv-team-box home">
                <div><div class="lv-team-name">{html.escape(home.get("name", ""))}</div>{h_rec}</div>
                {h_logo}
            </div>
            <div class="lv-score-pill" style="color:{score_cls};">{score_html}</div>
            <div class="lv-team-box away">
                {a_logo}
                <div><div class="lv-team-name">{html.escape(away.get("name", ""))}</div>{a_rec}</div>
            </div>
        </div>
        {ev_row_html}
        {ai_strip}
    </div>
    """), unsafe_allow_html=True)

    tab_evt, tab_xi, tab_comm, tab_form = st.tabs([
        "⚡ Diễn biến & Thống kê Live",
        "👕 Sơ đồ & Đội hình Live",
        "🎙️ Bình luận trực tiếp",
        "📊 Phong độ & Thông tin",
    ])

    with tab_evt:
        c_ke, c_st = st.columns([1.05, 1.15], gap="medium")
        with c_ke:
            st.markdown('<div style="color:#E2E8F0;font-weight:800;font-size:13px;margin-bottom:6px;">⏱️ Dòng thời gian trận đấu</div>', unsafe_allow_html=True)
            k_evs = list(summ.get("key_events") or [])
            if state == "in":
                k_evs = list(reversed(k_evs))
            if not k_evs:
                st.caption("Chưa có sự kiện chính nào được ghi nhận.")
            else:
                ev_items = []
                for ke in k_evs[:30]:
                    clk = html.escape(str(ke.get("clock") or "•"))
                    txt = html.escape(str(ke.get("text") or ke.get("type_text") or ""))
                    tm = html.escape(str(ke.get("team_name") or ""))
                    if ke.get("scoring_play"):
                        ico, bg_c = "⚽", "rgba(16,185,129,0.14)"
                    elif ke.get("is_red"):
                        ico, bg_c = "🟥", "rgba(239,68,68,0.16)"
                    elif ke.get("is_yellow"):
                        ico, bg_c = "🟨", "rgba(250,204,21,0.12)"
                    elif ke.get("is_sub"):
                        ico, bg_c = "🔄", "rgba(56,189,248,0.10)"
                    else:
                        ico, bg_c = "⏱️", "rgba(15,23,42,0.72)"
                    tm_tag = f' <span style="color:#7DD3FC;font-weight:700;">[{tm}]</span>' if tm else ""
                    ev_items.append(
                        f'<div class="lv-evt-item" style="background:{bg_c};">'
                        f'<span class="lv-evt-clk">{clk}</span>'
                        f'<span>{ico}{tm_tag} {txt}</span></div>'
                    )
                st.markdown(f'<div class="lv-comm-feed">{"".join(ev_items)}</div>', unsafe_allow_html=True)

        with c_st:
            st.markdown('<div style="color:#E2E8F0;font-weight:800;font-size:13px;margin-bottom:6px;">📈 Chỉ số trực tiếp trên sân</div>', unsafe_allow_html=True)
            stats = summ.get("stats") or []
            if not stats:
                st.caption("Chỉ số thống kê trực tiếp sẽ cập nhật ngay khi bóng lăn.")
            else:
                stat_rows = []
                for _k, lbl, h_s, a_s, h_n, a_n in stats:
                    tot = max(1e-6, h_n + a_n)
                    h_pct = (h_n / tot * 100.0) if (h_n + a_n) > 0 else 50.0
                    a_pct = 100.0 - h_pct
                    stat_rows.append(
                        f'<div class="lv-stat-row">'
                        f'<div class="lv-stat-top"><span style="color:#7DD3FC;font-weight:800;">{html.escape(h_s)}</span>'
                        f'<span style="color:#CBD5E1;font-weight:600;font-size:11.5px;">{html.escape(lbl)}</span>'
                        f'<span style="color:#FDA4AF;font-weight:800;">{html.escape(a_s)}</span></div>'
                        f'<div class="lv-stat-bar-wrap">'
                        f'<div style="width:{h_pct:.1f}%;background:linear-gradient(90deg,#0EA5E9,#38BDF8);height:100%;border-radius:4px 0 0 4px;"></div>'
                        f'<div style="width:{a_pct:.1f}%;background:linear-gradient(90deg,#F43F5E,#FB7185);height:100%;border-radius:0 4px 4px 0;"></div>'
                        f'</div></div>'
                    )
                st.markdown("".join(stat_rows), unsafe_allow_html=True)

    with tab_xi:
        rosters = summ.get("rosters") or []
        if len(rosters) < 2:
            st.info("Đội hình ra sân chính thức sẽ hiển thị ngay khi ESPN công bố (thường ~60 phút trước giờ bóng lăn).")
        else:
            h_ros, a_ros = rosters[0], rosters[1]
            if h_ros.get("starters") and a_ros.get("starters"):
                st.markdown(_build_live_tactical_pitch_svg(h_ros, a_ros), unsafe_allow_html=True)
            rc1, rc2 = st.columns(2, gap="medium")
            for col_r, ros, acc in ((rc1, h_ros, "#38BDF8"), (rc2, a_ros, "#FDA4AF")):
                with col_r:
                    st.markdown(f'<div style="color:{acc};font-weight:800;font-size:13px;margin:4px 0;">{html.escape(ros.get("team_name",""))} · {html.escape(ros.get("formation") or "Đội hình")}</div>', unsafe_allow_html=True)
                    for grp_lbl, arr in (("Đá chính", ros.get("starters") or []), ("Dự bị", ros.get("bench") or [])):
                        if not arr:
                            continue
                        rows_h = []
                        for p in arr:
                            badges = []
                            if p.get("goals"):
                                badges.append(f"⚽×{p['goals']}" if p['goals'] > 1 else "⚽")
                            if p.get("assists"):
                                badges.append("🅰️")
                            if p.get("yellow_cards"):
                                badges.append("🟨")
                            if p.get("red_cards"):
                                badges.append("🟥")
                            if p.get("subbed_in"):
                                badges.append("🔼Vào")
                            if p.get("subbed_out"):
                                badges.append("🔽Ra")
                            if p.get("saves"):
                                badges.append(f"🧤{p['saves']}")
                            bd_txt = " ".join(badges)
                            rows_h.append(
                                f'<div class="lv-roster-row">'
                                f'<span><b style="color:{acc};min-width:22px;display:inline-block;">#{html.escape(str(p.get("jersey") or "-"))}</b> '
                                f'{html.escape(str(p.get("name") or ""))} '
                                f'<span style="color:#64748B;font-size:10.5px;">({html.escape(str(p.get("pos") or ""))})</span></span>'
                                f'<span style="font-size:11px;white-space:nowrap;">{html.escape(bd_txt)}</span></div>'
                            )
                        st.markdown(f'<div style="color:#94A3B8;font-size:11px;font-weight:700;margin:6px 0 3px;">{grp_lbl}</div>{"".join(rows_h)}', unsafe_allow_html=True)

    with tab_comm:
        comm = list(summ.get("commentary") or [])
        c_mode = st.radio(
            "Lọc bình luận:",
            ["Tất cả diễn biến", "Điểm nóng (Bàn thắng / Sút / Thẻ)"],
            horizontal=True,
            key=f"lv_comm_flt_{event_id}",
        )
        if state == "in":
            comm = list(reversed(comm))
        if "Điểm nóng" in c_mode:
            comm = [
                c for c in comm
                if any(k in (c.get("text", "") + " " + c.get("type_text", "")).lower() for k in ("goal", "shot", "attempt", "saved", "miss", "card", "penalty", "post", "bar", "var"))
            ]
        if not comm:
            st.caption("Chưa có bình luận từng phút cho trận đấu này.")
        else:
            c_rows = []
            for cm in comm[:120]:
                clk = html.escape(str(cm.get("clock") or "•"))
                txt = html.escape(str(cm.get("text") or ""))
                c_rows.append(
                    f'<div class="lv-evt-item">'
                    f'<span class="lv-evt-clk">{clk}</span><span>{txt}</span></div>'
                )
            st.markdown(f'<div class="lv-comm-feed">{"".join(c_rows)}</div>', unsafe_allow_html=True)

    with tab_form:
        fc_a, fc_b = st.columns([1.2, 1.0], gap="medium")
        with fc_a:
            st.markdown('<div style="color:#E2E8F0;font-weight:800;font-size:13px;margin-bottom:6px;">🔥 Phong độ 5 trận gần nhất</div>', unsafe_allow_html=True)
            lf_list = summ.get("last_five") or []
            if not lf_list:
                st.caption("Chưa có dữ liệu 5 trận gần nhất từ ESPN.")
            else:
                for lf_t in lf_list:
                    st.markdown(f'<div style="color:#7DD3FC;font-weight:700;font-size:12px;margin:6px 0 3px;">{html.escape(lf_t.get("team_name",""))}</div>', unsafe_allow_html=True)
                    for ev in (lf_t.get("events") or []):
                        res_c = "#10B981" if ev.get("result") == "W" else ("#EF4444" if ev.get("result") == "L" else "#94A3B8")
                        st.markdown(
                            f'<div class="lv-roster-row">'
                            f'<span><b style="color:{res_c};display:inline-block;min-width:18px;">{html.escape(ev.get("result") or "-")}</b> '
                            f'{html.escape(ev.get("at_vs") or "")} {html.escape(ev.get("opponent") or "")}</span>'
                            f'<span style="color:#E2E8F0;font-weight:700;">{html.escape(ev.get("score") or "")} '
                            f'<span style="color:#64748B;font-size:10px;">{html.escape(ev.get("date") or "")}</span></span></div>',
                            unsafe_allow_html=True,
                        )
        with fc_b:
            st.markdown('<div style="color:#E2E8F0;font-weight:800;font-size:13px;margin-bottom:6px;">🏟️ Thông tin sân & Phát sóng</div>', unsafe_allow_html=True)
            info_lines = []
            if venue_txt:
                info_lines.append(f"• **Sân vận động:** {venue_txt}")
            if gi.get("attendance"):
                info_lines.append(f"• **Khán giả:** {gi['attendance']:,}" if isinstance(gi["attendance"], int) else f"• **Khán giả:** {gi['attendance']}")
            if gi.get("officials"):
                info_lines.append(f"• **Trọng tài:** {', '.join(gi['officials'][:3])}")
            if bcasts:
                info_lines.append(f"• **Kênh phát sóng (ESPN/TV):** {', '.join(bcasts)}")
            if summ.get("kickoff_vn"):
                info_lines.append(f"• **Giờ bóng lăn (VN):** {summ['kickoff_vn']}")
            st.markdown("\n".join(info_lines) if info_lines else "Đang cập nhật thông tin trận đấu.")

            vids = summ.get("videos") or []
            if vids:
                st.markdown('<div style="color:#E2E8F0;font-weight:800;font-size:13px;margin:10px 0 4px;">🎬 Video & Highlight (ESPN)</div>', unsafe_allow_html=True)
                for vd in vids[:4]:
                    st.markdown(f"- [{vd['headline']}]({vd['url']})")


@st.fragment(run_every=timedelta(seconds=30))
def _render_live_viewer_body_auto(slug: str, event_id: str, fallback_match: dict):
    force_once = bool(st.session_state.pop("daily_live_viewer_force", False))
    _render_live_viewer_body(slug, event_id, fallback_match, force_once)


def _render_inline_live_viewer(live_now: list[dict], viewer_cfg: dict):
    sel_eid = str(viewer_cfg.get("event_id") or "")
    sel_slug = str(viewer_cfg.get("slug") or "")
    fb_match = dict(viewer_cfg.get("match") or {})
    # Keep fallback_match fresh from latest scoreboard list if matching event_id is present
    for lm in live_now:
        if str(lm.get("espn_event_id") or "") == sel_eid:
            fb_match = lm
            break

    is_live = fb_match.get("state") == "in"
    with st.container(border=True):
        # Top control & live-switch bar
        hdr_title = f"📺 XEM TRỰC TIẾP TRÊN MÀN HÌNH · {fb_match.get('home_team', '')} vs {fb_match.get('away_team', '')}"
        st.markdown(f'<div style="color:#F8FAFC;font-weight:800;font-size:13.5px;margin-bottom:6px;">{html.escape(hdr_title)}</div>', unsafe_allow_html=True)

        if len(live_now) > 1:
            sw_cols = st.columns(min(len(live_now), 5), gap="small")
            for idx_s, lm in enumerate(live_now[:5]):
                l_eid = str(lm.get("espn_event_id") or "")
                sc_s = f"{lm.get('home_score')}-{lm.get('away_score')}" if lm.get("home_score") is not None else "vs"
                btn_lbl = f"{lm.get('home_team','')[:10]} {sc_s} {lm.get('away_team','')[:10]} ({lm.get('status_detail','')})"
                sw_cols[idx_s].button(
                    btn_lbl,
                    key=f"lv_quick_sw_{idx_s}_{l_eid}",
                    type="primary" if l_eid == sel_eid else "secondary",
                    width="stretch",
                    on_click=_open_inline_live_viewer,
                    args=(lm,),
                )

        c_auto, c_full, c_ref, c_ai, c_cls = st.columns([1.45, 1.30, 0.88, 0.98, 0.72], gap="small")
        auto_on = False
        if is_live:
            auto_on = c_auto.toggle("⏱️ Tự động 30s", value=True, key="daily_live_auto_30s")
        else:
            c_auto.caption("✅ Đã kết thúc — xem lại")
        if c_full.button("🏟️ Xem đầy đủ 6 tab", key="lv_expand_full_mc", type="primary", width="stretch"):
            _open_full_match_center(fb_match)
        if c_ref.button("🔄 Làm mới", key="lv_manual_refresh", width="stretch"):
            st.session_state["daily_live_viewer_force"] = True
        uid = str(fb_match.get("upcoming_match_id") or "")
        if c_ai.button("🔮 Soi kèo AI", key="lv_jump_ai", width="stretch"):
            if uid:
                st.session_state["jump_upcoming_match_id"] = uid
            else:
                st.session_state["jump_custom_home"] = str(fb_match.get("home_team") or "")
                st.session_state["jump_custom_away"] = str(fb_match.get("away_team") or "")
                st.session_state["prematch_tabs_sel"] = "⚡ Ghép 2 Đội"
            st.session_state["pending_workspace"] = "ws_prematch"
            st.rerun()
        c_cls.button("✖ Đóng", key="lv_close_btn", width="stretch", on_click=_close_inline_live_viewer)

        if auto_on and is_live:
            _render_live_viewer_body_auto(sel_slug, sel_eid, fb_match)
        else:
            force_once = bool(st.session_state.pop("daily_live_viewer_force", False))
            _render_live_viewer_body(sel_slug, sel_eid, fb_match, force_once)


def _open_full_match_center(m: dict):
    cat_ref = m.get("catalog_ref")
    if cat_ref and m.get("state") == "post":
        st.session_state.pop("daily_match_center_fixture", None)
        st.session_state["jump_comp"] = cat_ref[0]
        st.session_state["jump_match_label"] = cat_ref[1]
    else:
        st.session_state["daily_match_center_fixture"] = dict(m)
    st.session_state["return_to_ws_today"] = True
    st.session_state["pending_workspace"] = "ws_match"
    st.rerun()


def _jump_to_prematch(m: dict):
    uid = str(m.get("upcoming_match_id") or "")
    if uid:
        st.session_state["jump_upcoming_match_id"] = uid
    else:
        st.session_state["jump_custom_home"] = str(m.get("home_team") or "")
        st.session_state["jump_custom_away"] = str(m.get("away_team") or "")
        st.session_state["prematch_tabs_sel"] = "⚡ Ghép 2 Đội"
    st.session_state["pending_workspace"] = "ws_prematch"


def _render_match_row_card(m: dict, now_ts: float, btn_key: str, show_date_badge: bool = False):
    ht = str(m.get("home_team") or "")
    at = str(m.get("away_team") or "")
    state = m.get("state", "pre")
    vn_t = str(m.get("vn_time") or "")
    vn_d = str(m.get("vn_date") or "")

    # Time + Status Pill
    if state == "in":
        time_top = f"🔴 LIVE · {html.escape(str(m.get('status_detail') or 'Đang đá'))}"
        time_sub = vn_t or "Trực tiếp"
        pill_style = "background:rgba(239,68,68,0.18);border:1px solid rgba(239,68,68,0.5);color:#FCA5A5;"
        card_border = "1px solid rgba(239,68,68,0.42);border-left:3.5px solid #EF4444;"
    elif state == "post":
        time_top = f"✅ {html.escape(str(m.get('status_detail') or 'FT'))}"
        time_sub = f"{vn_d[8:10]}/{vn_d[5:7]} · {vn_t}" if show_date_badge and vn_t else (vn_t or "Đã kết thúc")
        pill_style = "background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.32);color:#34D399;"
        card_border = "1px solid rgba(148,163,184,0.15);border-left:3.5px solid #10B981;"
    else:
        time_top = (f"{vn_d[8:10]}/{vn_d[5:7]} · {vn_t}" if (show_date_badge and vn_t) else (vn_t or f"{vn_d[8:10]}/{vn_d[5:7]}" if len(vn_d) >= 10 else "Sắp đá"))
        time_sub = _format_relative_kickoff(m, now_ts)
        pill_style = "background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.34);color:#7DD3FC;"
        card_border = "1px solid rgba(148,163,184,0.16);border-left:3.5px solid #38BDF8;"

    # Center Score / VS
    hs = m.get("home_score")
    as_ = m.get("away_score")
    if hs is not None and as_ is not None:
        score_col = "#FCA5A5" if state == "in" else "#00E676"
        center_html = f'<span style="display:inline-flex;align-items:center;justify-content:center;font-size:17px;font-weight:900;color:{score_col};font-variant-numeric:tabular-nums;min-width:52px;text-align:center;">{int(hs)} - {int(as_)}</span>'
    else:
        center_html = '<span style="display:inline-flex;align-items:center;justify-content:center;min-width:52px;text-align:center;font-size:11.5px;font-weight:800;color:#7DD3FC;background:#090D16;border:1px solid rgba(56,189,248,0.24);padding:2px 8px;border-radius:6px;letter-spacing:0.06em;">VS</span>'

    # AI + Market Odds Compact Strip
    chips = []
    if m.get("prob_H") is not None:
        chips.append(
            f'<span style="color:#E2E8F0;">AI <b style="color:#7DD3FC;">{m["prob_H"]:.0f}%</b>·<b>{m["prob_D"]:.0f}%</b>·<b style="color:#FDA4AF;">{m["prob_A"]:.0f}%</b></span>'
        )
        chips.append(f'<span style="color:#94A3B8;">xG {m["pred_home_goals"]:.1f}-{m["pred_away_goals"]:.1f}</span>')
    elif m.get("hist_home_xg") is not None:
        chips.append(f'<span style="color:#94A3B8;">xG {m["hist_home_xg"]:.2f}-{m["hist_away_xg"]:.2f}</span>')
    od = m.get("odds") or {}
    if od.get("mkt_prob_H"):
        chips.append(f'<span style="color:#FACC15;">Kèo {od["mkt_prob_H"]:.0f}/{od["mkt_prob_D"]:.0f}/{od["mkt_prob_A"]:.0f}</span>')
    if not chips and m.get("venue"):
        chips.append(f'<span style="color:#64748B;">🏟️ {html.escape(str(m["venue"])[:26])}</span>')
    chips_html = " <span style='color:#334155;'>|</span> ".join(chips)

    h_badge = _team_badge_html(ht, m.get("home_logo", ""), is_home=True)
    a_badge = _team_badge_html(at, m.get("away_logo", ""), is_home=False)

    eid = str(m.get("espn_event_id") or "")
    slug = str(m.get("espn_slug") or "")
    has_espn = bool(eid and slug)
    cur_viewer = st.session_state.get("daily_live_viewer_event") or {}
    is_viewing = bool(eid and str(cur_viewer.get("event_id") or "") == eid)

    col_ratios = [4.30, 1.55]

    col_card, col_act = st.columns(col_ratios, gap="small")
    with col_card:
        st.markdown(compact_html(f"""
        <div class="dm-card dm-clickable" role="button" tabindex="0" title="Bấm trực tiếp vào trận đấu để xem chi tiết đầy đủ 6 tab thông số" style="border:{card_border};">
            <div class="dm-time">
                <span class="dm-time-pill" style="{pill_style}">{time_top}</span>
                <span class="dm-time-sub">{html.escape(time_sub)}</span>
            </div>
            <div class="dm-match">
                <div class="dm-team home">
                    <span class="dm-team-txt">{html.escape(ht)}</span>
                    {h_badge}
                </div>
                {center_html}
                <div class="dm-team away">
                    {a_badge}
                    <span class="dm-team-txt">{html.escape(at)}</span>
                </div>
            </div>
            <div class="dm-chips">{chips_html}</div>
            <span class="dm-chevron" aria-hidden="true">›</span>
        </div>
        """), unsafe_allow_html=True)

    with col_act:
        if state == "in" and has_espn:
            b_full, b_quick = st.columns([1.15, 0.85], gap="small")
            if b_full.button("🔴 Xem trận Live", key=btn_key, type="primary", width="stretch"):
                _open_full_match_center(m)
            b_quick.button(
                "🔴 Đang xem" if is_viewing else "⚡ Xem nhanh",
                key=f"{btn_key}_qk",
                help="Mở khung xem nhanh trực tiếp ngay bên dưới",
                width="stretch",
                on_click=_open_inline_live_viewer,
                args=(m,),
            )
        elif state == "pre":
            b_full, b_ai = st.columns([1.25, 0.75], gap="small")
            if b_full.button("🏟️ Xem trận", key=btn_key, type="primary", width="stretch"):
                _open_full_match_center(m)
            if b_ai.button("🔮 AI", key=f"{btn_key}_ai", help="Mở Trung tâm AI & Bộ quét kèo", width="stretch"):
                _jump_to_prematch(m)
                st.rerun()
        else:
            if st.button("🏟️ Xem trận", key=btn_key, type="primary", width="stretch"):
                _open_full_match_center(m)


@st.fragment
def _render_daily_schedule_fragment(fixtures: list[dict], updated_vn: str, ui_lang: str):
    now_vn = datetime.now(VN_TZ)
    today_date = now_vn.date()
    today_str = today_date.strftime("%Y-%m-%d")
    current_monday = today_date - timedelta(days=today_date.weekday())

    raw_wk_offsets = {0}
    for m in fixtures:
        d_o = _parse_iso_date(str(m.get("vn_date") or ""))
        if d_o is not None:
            wk_o = (d_o - current_monday).days // 7
            if -20 <= wk_o <= 32:
                raw_wk_offsets.add(wk_o)
    min_wk, max_wk = min(raw_wk_offsets), max(raw_wk_offsets)
    week_offsets = list(range(min_wk, max_wk + 1))

    if "daily_sel_date" not in st.session_state:
        st.session_state["daily_sel_date"] = today_str
    sel_date = str(st.session_state["daily_sel_date"])
    if "daily_date_picker" not in st.session_state:
        st.session_state["daily_date_picker"] = _parse_iso_date(sel_date) or today_date

    def _on_picker_change():
        picked = st.session_state.get("daily_date_picker")
        if picked is not None:
            st.session_state["daily_sel_date"] = picked.strftime("%Y-%m-%d")
            wk = max(min_wk, min(max_wk, (picked - current_monday).days // 7))
            st.session_state["daily_week_slider"] = wk

    # Top filter & sync bar
    avail_lgs = [lg for lg in _EXTENDED_LEAGUE_ORDER if any(m.get("league_code") == lg for m in fixtures)]
    lg_opts = ["ALL", "TOP7"] + avail_lgs

    def _fmt_lg(lg):
        if lg == "ALL":
            return "🌍 Tất cả 13 giải đấu"
        if lg == "TOP7":
            return "🌟 Top 5 Châu Âu + UCL/UEL"
        if lg == "INTL_FR":
            return "🌍 Giao Hữu Quốc Tế"
        return LEAGUE_META.get(lg, (lg, lg))[0]

    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.4, 1.0, 1.1], gap="small")
    sel_lg = fc1.selectbox("Giải đấu:", lg_opts, index=0, format_func=_fmt_lg, key="daily_lg_flt")
    q_team = fc2.text_input("Tìm đội bóng:", placeholder="VD: Liverpool, Real Madrid, Arsenal...", key="daily_team_q")
    fc3.date_input("Chọn ngày khác:", key="daily_date_picker", on_change=_on_picker_change)
    with fc4:
        st.markdown('<div style="height:26px;"></div>', unsafe_allow_html=True)
        if st.button("🔄 Cập nhật Live", width="stretch", key="daily_force_refresh_btn"):
            res = get_realtime_board_data(force_refresh=True)
            if res.get("cooldown_wait"):
                st.toast(f"Dữ liệu vừa cập nhật xong (thử lại sau {res['cooldown_wait']}s).")
            st.rerun()

    sel_date = str(st.session_state["daily_sel_date"])

    # Filter fixtures by league and search text
    q_low = str(q_team or "").strip().lower()
    flt_fixtures = []
    for m in fixtures:
        lg = str(m.get("league_code") or "")
        if sel_lg == "TOP7" and lg not in ("ENG", "ESP", "ITA", "GER", "FRA", "UCL", "UEL"):
            continue
        elif sel_lg not in ("ALL", "TOP7") and lg != sel_lg:
            continue
        if q_low and q_low not in f"{m.get('home_team', '')} {m.get('away_team', '')} {m.get('competition', '')}".lower():
            continue
        flt_fixtures.append(m)

    counts_by_date: dict[str, int] = {}
    counts_by_week: dict[int, int] = {}
    for m in flt_fixtures:
        d_k = str(m.get("vn_date") or "")[:10]
        d_o = _parse_iso_date(d_k)
        if d_o is not None:
            counts_by_date[d_k] = counts_by_date.get(d_k, 0) + 1
            wk_k = (d_o - current_monday).days // 7
            counts_by_week[wk_k] = counts_by_week.get(wk_k, 0) + 1

    def _default_date_for_week(wk_off: int) -> str:
        if wk_off == 0:
            return today_str
        ws = current_monday + timedelta(days=wk_off * 7)
        for idx_day in range(7):
            cand = (ws + timedelta(days=idx_day)).strftime("%Y-%m-%d")
            if counts_by_date.get(cand, 0) > 0:
                return cand
        return ws.strftime("%Y-%m-%d")

    def _pick_date_btn(target_d_str: str):
        st.session_state["daily_sel_date"] = target_d_str
        dt_o = _parse_iso_date(target_d_str) or today_date
        st.session_state["daily_date_picker"] = dt_o
        st.session_state["daily_week_slider"] = max(min_wk, min(max_wk, (dt_o - current_monday).days // 7))

    def _on_week_slider_change():
        wk_off = int(st.session_state.get("daily_week_slider", 0))
        target_d_str = _default_date_for_week(wk_off)
        st.session_state["daily_sel_date"] = target_d_str
        st.session_state["daily_date_picker"] = _parse_iso_date(target_d_str) or today_date

    def _jump_to_week(target_wk: int):
        wk_c = max(min_wk, min(max_wk, int(target_wk)))
        st.session_state["daily_week_slider"] = wk_c
        target_d_str = _default_date_for_week(wk_c)
        st.session_state["daily_sel_date"] = target_d_str
        st.session_state["daily_date_picker"] = _parse_iso_date(target_d_str) or today_date

    sel_dt_obj = _parse_iso_date(sel_date) or today_date
    cur_wk = max(min_wk, min(max_wk, (sel_dt_obj - current_monday).days // 7))
    if st.session_state.get("daily_week_slider") not in week_offsets or st.session_state.get("daily_week_slider") != cur_wk:
        st.session_state["daily_week_slider"] = cur_wk

    w_start = current_monday + timedelta(days=cur_wk * 7)
    w_end = w_start + timedelta(days=6)
    wk_total = counts_by_week.get(cur_wk, 0)

    def _fmt_week_opt(wk_off: int) -> str:
        ws = current_monday + timedelta(days=wk_off * 7)
        we = ws + timedelta(days=6)
        if wk_off == 0:
            tag = "Tuần này"
        elif wk_off == 1:
            tag = "Tuần tới"
        elif wk_off == -1:
            tag = "Tuần trước"
        elif wk_off > 0:
            tag = f"+{wk_off} tuần"
        else:
            tag = f"{wk_off} tuần"
        return f"{tag} ({ws.strftime('%d/%m')}–{we.strftime('%d/%m')})"

    # Interactive Week Slider Bar
    st.markdown(
        f'<div class="wk-bar-hdr"><span>📆 Chuyển tuần nhanh · <b style="color:#7DD3FC;">Tuần {w_start.strftime("%d/%m")} – {w_end.strftime("%d/%m/%Y")}</b> ({wk_total} trận)</span>'
        f'<span style="color:#64748B;font-size:11px;">Kéo thanh trượt hoặc bấm ◀ / ▶ để xem các tuần trước và sắp tới</span></div>',
        unsafe_allow_html=True,
    )
    wc1, wc2, wc3, wc4 = st.columns([0.92, 3.8, 0.92, 0.86], gap="small")
    wc1.button("◀ Tuần trước", key="daily_wk_prev", disabled=(cur_wk <= min_wk), width="stretch", on_click=_jump_to_week, args=(cur_wk - 1,))
    wc2.select_slider(
        "Chuyển tuần",
        options=week_offsets,
        format_func=_fmt_week_opt,
        key="daily_week_slider",
        label_visibility="collapsed",
        on_change=_on_week_slider_change,
    )
    wc3.button("Tuần sau ▶", key="daily_wk_next", disabled=(cur_wk >= max_wk), width="stretch", on_click=_jump_to_week, args=(cur_wk + 1,))
    wc4.button("📍 Hôm nay", key="daily_wk_today", disabled=(sel_date == today_str), width="stretch", on_click=_pick_date_btn, args=(today_str,))

    # Dynamic 7-Day Interactive Strip (Mon..Sun of selected week)
    day_cols = st.columns(7, gap="small")
    for idx_d in range(7):
        d_obj = w_start + timedelta(days=idx_d)
        d_s = d_obj.strftime("%Y-%m-%d")
        cnt = counts_by_date.get(d_s, 0)
        delta_today = (d_obj - today_date).days
        if delta_today == 0:
            day_tag = "Hôm nay"
        elif delta_today == -1:
            day_tag = "Hôm qua"
        elif delta_today == 1:
            day_tag = "Ngày mai"
        else:
            day_tag = _VI_WEEKDAYS[d_obj.weekday()]
        lbl = f"{day_tag} · {d_obj.strftime('%d/%m')} ({cnt})"
        day_cols[idx_d].button(
            lbl,
            key=f"day_strip_{d_s}",
            type="primary" if d_s == sel_date else "secondary",
            width="stretch",
            on_click=_pick_date_btn,
            args=(d_s,),
        )

    now_ts = datetime.now(timezone.utc).timestamp()
    live_now = [m for m in flt_fixtures if m.get("state") == "in"]
    live_now.sort(key=_sort_match_key)

    # Inline Live Match Viewer Panel (when user clicks "📺 Xem Live" or "📺 Diễn biến")
    viewer_cfg = st.session_state.get("daily_live_viewer_event")
    if isinstance(viewer_cfg, dict) and viewer_cfg.get("event_id") and viewer_cfg.get("slug"):
        _render_inline_live_viewer(live_now, viewer_cfg)

    # Pinned Live Now Strip on Hôm nay (or 1-click Live banner on other dates/weeks)
    if live_now:
        if sel_date == today_str:
            st.markdown(
                f'<div class="section-hdr" style="margin-top:8px;"><span class="section-hdr-title" style="color:#FCA5A5;">🔴 ĐANG DIỄN RA TRỰC TIẾP ({len(live_now)})</span>'
                f'<span class="section-hdr-sub">Bấm trực tiếp vào bất kỳ trận đấu nào để xem đầy đủ 6 tab thông số · hoặc ⚡ Xem nhanh tại đây</span></div>',
                unsafe_allow_html=True,
            )
            for i_l, lm in enumerate(live_now):
                _render_match_row_card(lm, now_ts, btn_key=f"live_pin_{i_l}_{lm.get('espn_event_id')}_{lm.get('home_team')}", show_date_badge=False)
        else:
            live_snip = ", ".join(
                f"{x.get('home_team', '')} {x.get('home_score')}-{x.get('away_score')} {x.get('away_team', '')}"
                for x in live_now[:2]
            )
            ellipsis = "…" if len(live_now) > 2 else ""
            lb1, lb2 = st.columns([4.2, 1.4], gap="small")
            lb1.markdown(
                f'<div style="padding:7px 12px;border-radius:9px;background:rgba(239,68,68,0.10);border:1px solid rgba(239,68,68,0.35);color:#FCA5A5;font-size:12.5px;font-weight:700;">'
                f'🔴 Hiện có <b>{len(live_now)}</b> trận đang diễn ra trực tiếp hôm nay ({html.escape(live_snip)}{ellipsis})'
                f'</div>',
                unsafe_allow_html=True,
            )
            lb2.button(
                "📺 Xem trực tiếp ngay",
                key="live_banner_jump_watch",
                type="primary",
                width="stretch",
                on_click=_open_inline_live_viewer,
                args=(live_now[0],),
            )

    # Matches on selected date grouped by competition
    day_matches = [m for m in flt_fixtures if str(m.get("vn_date") or "") == sel_date]
    day_matches.sort(key=_sort_match_key)

    sel_disp = f"{_VI_WEEKDAYS[sel_dt_obj.weekday()]}, {sel_dt_obj.strftime('%d/%m/%Y')}"

    st.markdown(
        f'<div class="section-hdr" style="margin-top:10px;"><span class="section-hdr-title">📅 Lịch Thi Đấu Ngày {html.escape(sel_disp)} ({len(day_matches)} trận)</span>'
        f'<span class="section-hdr-sub">Giờ thi đấu quy đổi GMT+7 · Bấm trực tiếp vào bất kỳ trận đấu nào để xem chi tiết đầy đủ 6 tab thông số</span></div>',
        unsafe_allow_html=True,
    )

    show_next = []
    if not day_matches:
        st.info(f"Không có trận đấu nào khớp bộ lọc vào ngày **{sel_disp}**. Hãy bấm các ngày có số trận (>0) trên thanh tuần hoặc kéo thanh trượt sang tuần khác!")
    else:
        grouped: dict[str, list[dict]] = {}
        for m in day_matches:
            grouped.setdefault(str(m.get("league_code") or "OTHER"), []).append(m)

        ordered_codes = [c for c in _EXTENDED_LEAGUE_ORDER if c in grouped] + [c for c in grouped if c not in _EXTENDED_LEAGUE_ORDER]
        row_counter = 0
        for lg_code in ordered_codes:
            items = grouped[lg_code]
            items.sort(key=_sort_match_key)
            comp_hdr = str(items[0].get("competition") or lg_code)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin:10px 0 6px;padding:5px 11px;border-radius:8px;'
                f'background:rgba(15,23,42,0.85);border:1px solid rgba(148,163,184,0.14);">'
                f'<span style="color:#E2E8F0;font-weight:800;font-size:12.5px;">{html.escape(comp_hdr)}</span>'
                f'<span style="color:#64748B;font-size:11px;font-weight:700;">{len(items)} trận</span></div>',
                unsafe_allow_html=True,
            )
            for m in items:
                row_counter += 1
                uid_tag = m.get("upcoming_match_id") or m.get("espn_event_id") or row_counter
                _render_match_row_card(m, now_ts, btn_key=f"day_{sel_date}_{lg_code}_{row_counter}_{uid_tag}", show_date_badge=False)

    # Smart "Chuẩn bị diễn ra tiếp theo" section when Today has < 6 remaining pre-match games
    today_pre_left = [m for m in day_matches if m.get("state") == "pre"]
    if sel_date == today_str and len(today_pre_left) < 6:
        next_upcoming = [
            m for m in flt_fixtures
            if m.get("state") == "pre" and str(m.get("vn_date") or "") > today_str
        ]
        next_upcoming.sort(key=_sort_match_key)
        if next_upcoming:
            show_next = next_upcoming[:8]
            st.markdown(
                f'<div class="section-hdr" style="margin-top:16px;"><span class="section-hdr-title">⏰ Chuẩn Bị Diễn Ra Tiếp Theo (48–72 giờ tới)</span>'
                f'<span class="section-hdr-sub">Tâm điểm Champions League & Top Giải Đấu tiếp nối ngay sau hôm nay</span></div>',
                unsafe_allow_html=True,
            )
            for i_n, nm in enumerate(show_next):
                uid_tag = nm.get("upcoming_match_id") or nm.get("espn_event_id") or i_n
                _render_match_row_card(nm, now_ts, btn_key=f"next_up_{i_n}_{uid_tag}", show_date_badge=True)

    # Save deduplicated playlist for 1-click switching across matches inside ws_match
    day_playlist = []
    seen_pl = set()
    for pm in (live_now + day_matches + show_next):
        pk = (
            str(pm.get("espn_event_id") or ""),
            str(pm.get("upcoming_match_id") or ""),
            str(pm.get("home_team") or ""),
            str(pm.get("away_team") or ""),
            str(pm.get("vn_date") or ""),
        )
        if pk not in seen_pl:
            seen_pl.add(pk)
            day_playlist.append(pm)
        if len(day_playlist) >= 30:
            break
    st.session_state["daily_day_playlist"] = day_playlist


def render_daily_realtime_home(catalog: dict, ui_lang: str = "vi"):
    """Entry point cho Không gian mặc định: Đồng hồ thời gian thực GMT+7 + Thanh trượt tuần + Xem Live tại chỗ."""
    live_data = get_realtime_board_data(force_refresh=False)
    up_df = load_upcoming_fixtures_df()
    hist_df = load_historical_xg_df()
    fixtures = build_merged_daily_fixtures(live_data, up_df, catalog, hist_df=hist_df)

    today_str = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    live_cnt = sum(1 for m in fixtures if m.get("state") == "in")
    today_pre_cnt = sum(1 for m in fixtures if m.get("vn_date") == today_str and m.get("state") == "pre")
    today_post_cnt = sum(1 for m in fixtures if m.get("vn_date") == today_str and m.get("state") == "post")

    _render_realtime_clock_banner(
        live_cnt=live_cnt,
        today_pre_cnt=today_pre_cnt,
        today_post_cnt=today_post_cnt,
        updated_vn=live_data.get("updated_vn", ""),
    )
    _render_daily_schedule_fragment(fixtures, live_data.get("updated_vn", ""), ui_lang)
