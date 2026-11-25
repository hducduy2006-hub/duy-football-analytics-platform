"""Team & Player Search Engine and Profile Explorer (SofaScore / FotMob style)."""

from __future__ import annotations

import gc
import html
import json
import os
import unicodedata as _ud
from pathlib import Path

import pandas as pd
import pyarrow.parquet as _pq
import plotly.graph_objects as go
import streamlit as st

from player_avatar_store import get_player_avatar_b64, make_broadcast_bust_svg
from prematch_research import (
    compact_html,
    dixon_coles_matrix,
    extract_team_recent_form,
    load_historical_xg_df,
    load_upcoming_fixtures_df,
)
from pro_features import compute_player_radar_metrics, load_fifa_transfer_db, lookup_fifa_player
from team_logos_store import get_team_logo_url

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"

_CHAR_TRANSLATE = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "ł": "l",
        "Ł": "l",
        "đ": "d",
        "Đ": "d",
        "ð": "d",
        "Ð": "d",
        "ß": "ss",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
    }
)

TEAM_SEARCH_ALIASES: dict[str, str] = {
    "barca": "barcelona",
    "fcb": "barcelona",
    "real": "real madrid",
    "rm": "real madrid",
    "mu": "manchester united",
    "man utd": "manchester united",
    "man u": "manchester united",
    "mc": "manchester city",
    "man city": "manchester city",
    "psg": "paris saint-germain",
    "paris": "paris saint-germain",
    "inter": "inter milan",
    "acm": "ac milan",
    "milan": "ac milan",
    "atleti": "atletico madrid",
    "juve": "juventus",
    "bayern": "bayern munich",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "bvb": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "leverkusen": "bayer leverkusen",
    "napoli": "napoli",
    "viet nam": "vietnam",
}


def _norm_key(text: str) -> str:
    raw = str(text or "").translate(_CHAR_TRANSLATE)
    decomposed = "".join(c for c in _ud.normalize("NFKD", raw) if not _ud.combining(c))
    cleaned = "".join(ch if ch.isalnum() else " " for ch in decomposed.lower())
    return " ".join(cleaned.split())


def _safe_int(val, default: int = 0) -> int:
    try:
        if val is None or pd.isna(val):
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _team_crest_img_html(team_name: str, size: int = 28) -> str:
    url = get_team_logo_url(team_name)
    if url:
        safe_u = html.escape(url, quote=True)
        safe_alt = html.escape(team_name, quote=True)
        return (
            f'<img src="{safe_u}" alt="{safe_alt}" '
            f'style="width:{size}px;height:{size}px;object-fit:contain;vertical-align:middle;flex-shrink:0;" />'
        )
    initials = "".join(w[0] for w in str(team_name or "FC").split()[:2]).upper() or "FC"
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:{size}px;height:{size}px;border-radius:50%;background:#1E293B;'
        f'color:#94A3B8;font-size:{max(9, size // 3)}px;font-weight:800;flex-shrink:0;">'
        f"{html.escape(initials)}</span>"
    )


def _player_portrait_html(name: str, club: str, size: int = 56, allow_network: bool = False) -> str:
    uri = ""
    try:
        uri = get_player_avatar_b64(name, team_name=club, allow_network=allow_network)
    except Exception:
        uri = ""
    if uri:
        return (
            f'<img src="{uri}" alt="{html.escape(name, quote=True)}" '
            f'style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;'
            f'border:2px solid rgba(56,189,248,0.45);flex-shrink:0;" />'
        )
    parts = str(name or "?").split()
    initials = ((parts[0][0] if parts else "") + (parts[-1][0] if len(parts) > 1 else "")).upper() or "?"
    return make_broadcast_bust_svg(initials, "#38BDF8", "#1E293B", size=size)


def _build_catalog_maps(catalog: dict) -> dict:
    """Summarize teams and Opta match-id index cleanly from current catalog."""
    total_matches = sum(len(v) for v in catalog.values())
    cached = st.session_state.get("_explorer_catalog_cache")
    if isinstance(cached, dict) and cached.get("total") == total_matches:
        return cached

    teams: dict[str, dict] = {}
    by_opta_id: dict[str, tuple[str, dict]] = {}

    for comp_name, mlist in catalog.items():
        for m in mlist:
            oid = str(m.get("opta_match_id") or "").strip()
            if oid and oid not in by_opta_id:
                by_opta_id[oid] = (comp_name, m)

            is_up = bool(m.get("is_upcoming"))
            for side_key in ("home_team", "away_team"):
                tname = str(m.get(side_key) or "").strip()
                if not tname or tname.lower() in ("tbd", "home", "away"):
                    continue
                tk = _norm_key(tname)
                if not tk:
                    continue
                rec = teams.get(tk)
                if rec is None:
                    rec = {
                        "key": tk,
                        "name": tname,
                        "played": 0,
                        "upcoming": 0,
                        "comps": set(),
                        "matches": [],
                    }
                    teams[tk] = rec
                if len(tname) > len(rec["name"]) and "fc" not in tname.lower():
                    rec["name"] = tname
                if is_up:
                    rec["upcoming"] += 1
                else:
                    rec["played"] += 1
                rec["comps"].add(comp_name)
                rec["matches"].append((comp_name, m))

    res = {"total": total_matches, "teams": teams, "by_opta_id": by_opta_id}
    st.session_state["_explorer_catalog_cache"] = res
    return res


def search_teams(catalog: dict, query: str, limit: int = 8) -> list[dict]:
    """Search clubs & national teams across all 7,100+ catalog matches."""
    qn = _norm_key(query)
    if not qn:
        return []
    qn_clean = _clean_club_key(qn) or qn
    alias_target = TEAM_SEARCH_ALIASES.get(qn, "") or TEAM_SEARCH_ALIASES.get(qn_clean, "")
    q_tokens = qn_clean.split()

    teams_map = _build_catalog_maps(catalog)["teams"]
    scored: list[tuple[tuple, dict]] = []

    for tk, rec in teams_map.items():
        tk_clean = _clean_club_key(tk) or tk
        total = rec["played"] + rec["upcoming"]
        if tk == qn or tk_clean == qn_clean or (alias_target and (tk == alias_target or tk_clean == alias_target)):
            tier = 0
        elif alias_target and alias_target in tk:
            tier = 1
        elif tk.startswith(qn_clean) or tk_clean.startswith(qn_clean):
            tier = 2
        elif any(w.startswith(qn_clean) for w in tk.split()):
            tier = 3
        elif qn_clean in tk or (len(q_tokens) > 1 and all(tok in tk for tok in q_tokens)):
            tier = 4
        else:
            continue

        scored.append(
            (
                (tier, -total, -rec["upcoming"], len(tk)),
                {
                    "name": rec["name"],
                    "key": tk,
                    "played": rec["played"],
                    "upcoming": rec["upcoming"],
                    "total": total,
                    "comps": sorted(rec["comps"]),
                },
            )
        )

    scored.sort(key=lambda item: item[0])
    out = []
    for _, item in scored[:limit]:
        item["logo_url"] = get_team_logo_url(item["name"])
        out.append(item)
    return out


def _clean_club_key(club: str) -> str:
    k = _norm_key(club)
    for prefix in ("fc ", "cf ", "ac ", "as ", "sc ", "ssc ", "rc ", "rcd ", "cd ", "ud ", "ca "):
        if k.startswith(prefix):
            k = k[len(prefix) :]
    for suffix in (" fc", " cf", " sc", " ac"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
    return k.strip()


def _load_lineup_aggregates() -> tuple[dict, list[dict]]:
    """Stream minimal columns from Opta 2025/27 + UCL lineups to keep RAM lean."""
    use_cols = [
        "match_id",
        "match_date",
        "player_name",
        "team_name",
        "position",
        "shirt_number",
        "is_starter",
        "minutes_played",
        "competition",
        "season",
    ]
    agg_map: dict[tuple[str, str], dict] = {}

    for rel in ("opta_modern/modern_lineups.parquet", "opta_ucl/ucl_lineups.parquet"):
        pq_path = SAMPLE_DIR / rel
        if not pq_path.is_file():
            continue
        try:
            pf = _pq.ParquetFile(pq_path)
            avail = set(pf.schema_arrow.names)
            cols = [c for c in use_cols if c in avail]
            for batch in pf.iter_batches(batch_size=32768, columns=cols):
                bdict = batch.to_pydict()
                n_rows = len(bdict.get("player_name") or [])
                pnames = bdict.get("player_name") or [""] * n_rows
                tnames = bdict.get("team_name") or [""] * n_rows
                mdates = bdict.get("match_date") or [""] * n_rows
                mids = bdict.get("match_id") or [""] * n_rows
                positions = bdict.get("position") or [""] * n_rows
                shirts = bdict.get("shirt_number") or [0] * n_rows
                starters = bdict.get("is_starter") or [False] * n_rows
                mins_col = bdict.get("minutes_played") or [0] * n_rows
                comps = bdict.get("competition") or [""] * n_rows
                seasons = bdict.get("season") or [""] * n_rows

                for i in range(n_rows):
                    pname = str(pnames[i] or "").strip()
                    tname = str(tnames[i] or "").strip()
                    if not pname or pname.lower() == "nan":
                        continue
                    pnorm = _norm_key(pname)
                    cnorm = _clean_club_key(tname)
                    key = (pnorm, cnorm)

                    is_st = bool(starters[i])
                    mins = _safe_int(mins_col[i], 0)
                    shirt = _safe_int(shirts[i], 0)
                    pos = str(positions[i] or "").strip()
                    mdate = str(mdates[i] or "")[:10]
                    mid = str(mids[i] or "")
                    comp = str(comps[i] or "").strip()
                    season = str(seasons[i] or "").strip()

                    rec = agg_map.get(key)
                    if rec is None:
                        rec = {
                            "player_name": pname,
                            "pnorm": pnorm,
                            "team_name": tname,
                            "cnorm": cnorm,
                            "position": pos if pos.lower() != "nan" else "",
                            "shirt": shirt,
                            "apps": 0,
                            "starts": 0,
                            "minutes": 0,
                            "latest_date": mdate,
                            "seasons": {},
                            "matches": [],
                            "_seen_mids": set(),
                        }
                        agg_map[key] = rec

                    if mid and mid in rec["_seen_mids"]:
                        continue
                    if mid:
                        rec["_seen_mids"].add(mid)

                    rec["apps"] += 1
                    if is_st:
                        rec["starts"] += 1
                    rec["minutes"] += mins
                    if mdate >= rec["latest_date"]:
                        rec["latest_date"] = mdate
                        if shirt > 0:
                            rec["shirt"] = shirt
                        if pos and pos.lower() != "nan":
                            rec["position"] = pos

                    sk = (season or "2025/26", comp or "Club", tname)
                    s_row = rec["seasons"].setdefault(sk, [0, 0, 0])
                    s_row[0] += 1
                    if is_st:
                        s_row[1] += 1
                    s_row[2] += mins

                    if mid:
                        rec["matches"].append((mdate, mid, comp, season, tname, is_st, mins))
        except Exception:
            continue

    agg_list = list(agg_map.values())
    for r in agg_list:
        r.pop("_seen_mids", None)
        r["matches"].sort(key=lambda x: x[0], reverse=True)
        if len(r["matches"]) > 20:
            r["matches"] = r["matches"][:20]
    gc.collect()
    return agg_map, agg_list


@st.cache_resource(show_spinner=False)
def get_player_search_index() -> dict:
    """Lazy unified player index across EA FC 26/27, Opta 25/27 + UCL, and FIFA DB."""
    players: list[dict] = []
    by_id: dict[str, dict] = {}
    by_full_club: dict[tuple[str, str], dict] = {}
    by_init_sur_club: dict[tuple[str, str, str], dict] = {}
    by_sur_club: dict[tuple[str, str], dict] = {}
    by_full: dict[str, dict] = {}

    def _register_keys(rec: dict, alias_name: str = ""):
        cnorm = rec["club_key"]
        for nm in (rec["name"], rec.get("common_name", ""), alias_name):
            nk = _norm_key(nm)
            if not nk:
                continue
            if nk not in rec["search_keys"]:
                rec["search_keys"].append(nk)
            if cnorm and (nk, cnorm) not in by_full_club:
                by_full_club[(nk, cnorm)] = rec
            if nk not in by_full:
                by_full[nk] = rec
            parts = nk.split()
            if len(parts) >= 2:
                sur = parts[-1]
                init = parts[0][0]
                if cnorm and (init, sur, cnorm) not in by_init_sur_club:
                    by_init_sur_club[(init, sur, cnorm)] = rec
                if cnorm and (sur, cnorm) not in by_sur_club:
                    by_sur_club[(sur, cnorm)] = rec

    # 1. EA FC 26/27 Catalog (16,228 players)
    ea_path = SAMPLE_DIR / "pro_analytics" / "ea_fc_portraits_cache.json"
    if ea_path.is_file():
        try:
            with open(ea_path, "r", encoding="utf-8") as f:
                ea_raw = json.load(f)
            for row in (ea_raw.get("players") or []):
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                pid, first, last, common, club, ovr = row[:6]
                full_name = f"{first} {last}".strip()
                disp = str(common or "").strip() or full_name
                if not disp:
                    continue
                club_s = str(club or "").strip()
                rec_id = f"ea:{pid}"
                rec = {
                    "id": rec_id,
                    "name": disp,
                    "full_name": full_name or disp,
                    "common_name": str(common or "").strip(),
                    "club": club_s,
                    "club_key": _clean_club_key(club_s),
                    "ovr": _safe_int(ovr, 70),
                    "position": "",
                    "shirt": 0,
                    "nationality": "",
                    "age": 0,
                    "value": "",
                    "wage": "",
                    "apps": 0,
                    "starts": 0,
                    "minutes": 0,
                    "seasons": [],
                    "matches": [],
                    "search_keys": [],
                    "source": "EA FC 26/27",
                }
                players.append(rec)
                by_id[rec_id] = rec
                _register_keys(rec, full_name)
        except Exception:
            pass

    # 2. Merge Opta Modern + UCL Lineups
    _, opta_list = _load_lineup_aggregates()
    for orec in opta_list:
        pnorm = orec["pnorm"]
        cnorm = orec["cnorm"]
        target = by_full_club.get((pnorm, cnorm))
        if target is None:
            parts = pnorm.split()
            if len(parts) >= 2:
                target = by_init_sur_club.get((parts[0][0], parts[-1], cnorm))
            elif parts:
                target = by_sur_club.get((parts[-1], cnorm))
        if target is None and len(pnorm.split()) >= 2 and len(pnorm.split()[0]) > 1:
            target = by_full.get(pnorm)

        season_rows = [
            {
                "season": sk[0],
                "competition": sk[1],
                "club": sk[2],
                "apps": vals[0],
                "starts": vals[1],
                "minutes": vals[2],
            }
            for sk, vals in sorted(orec["seasons"].items(), key=lambda kv: (kv[0][0], kv[1][0]), reverse=True)
        ]

        if target is not None:
            target["apps"] += orec["apps"]
            target["starts"] += orec["starts"]
            target["minutes"] += orec["minutes"]
            if orec["position"] and not target["position"]:
                target["position"] = orec["position"]
            if orec["shirt"] and not target["shirt"]:
                target["shirt"] = orec["shirt"]
            if orec["player_name"] and _norm_key(orec["player_name"]) not in target["search_keys"]:
                target["search_keys"].append(_norm_key(orec["player_name"]))
            target["seasons"].extend(season_rows)
            existing_mids = {m[1] for m in target["matches"]}
            for mrow in orec["matches"]:
                if mrow[1] not in existing_mids and len(target["matches"]) < 25:
                    target["matches"].append(mrow)
                    existing_mids.add(mrow[1])
            target["matches"].sort(key=lambda x: x[0], reverse=True)
        else:
            rec_id = f"opta:{pnorm}:{cnorm}"
            if rec_id in by_id:
                continue
            rec = {
                "id": rec_id,
                "name": orec["player_name"],
                "full_name": orec["player_name"],
                "common_name": "",
                "club": orec["team_name"],
                "club_key": cnorm,
                "ovr": 72 if orec["apps"] >= 5 else 68,
                "position": orec["position"],
                "shirt": orec["shirt"],
                "nationality": "",
                "age": 0,
                "value": "",
                "wage": "",
                "apps": orec["apps"],
                "starts": orec["starts"],
                "minutes": orec["minutes"],
                "seasons": season_rows,
                "matches": list(orec["matches"]),
                "search_keys": [pnorm],
                "source": "Opta 25/27 & UCL",
            }
            players.append(rec)
            by_id[rec_id] = rec
            _register_keys(rec)

    # 3. Enrich & add legends from FIFA DB
    try:
        fifa_df, _, _ = load_fifa_transfer_db()
    except Exception:
        fifa_df = pd.DataFrame()

    if not fifa_df.empty:
        col_names = fifa_df["Name"].fillna("").astype(str).tolist()
        col_clubs = fifa_df["Club"].fillna("").astype(str).tolist() if "Club" in fifa_df.columns else [""] * len(col_names)
        col_ovrs = fifa_df["Overall"].fillna(70).tolist() if "Overall" in fifa_df.columns else [70] * len(col_names)
        col_pos = fifa_df["Preferred Positions"].fillna("").astype(str).tolist() if "Preferred Positions" in fifa_df.columns else [""] * len(col_names)
        col_nats = fifa_df["Nationality"].fillna("").astype(str).tolist() if "Nationality" in fifa_df.columns else [""] * len(col_names)
        col_ages = fifa_df["Age"].fillna(0).tolist() if "Age" in fifa_df.columns else [0] * len(col_names)
        col_vals = fifa_df["Value"].fillna("").astype(str).tolist() if "Value" in fifa_df.columns else [""] * len(col_names)
        col_wages = fifa_df["Wage"].fillna("").astype(str).tolist() if "Wage" in fifa_df.columns else [""] * len(col_names)

        for idx, (fnm_raw, fclub_raw, fovr_raw, fpos_raw, fnat_raw, fage_raw, fval_raw, fwage_raw) in enumerate(
            zip(col_names, col_clubs, col_ovrs, col_pos, col_nats, col_ages, col_vals, col_wages)
        ):
            fnm = fnm_raw.strip()
            fnk = _norm_key(fnm)
            if not fnk:
                continue
            fclub = fclub_raw.strip()
            fcnorm = _clean_club_key(fclub)
            fovr = _safe_int(fovr_raw, 70)
            fpos = fpos_raw.strip()
            fnat = fnat_raw.strip()
            fage = _safe_int(fage_raw, 0)
            fval = fval_raw.strip()
            fwage = fwage_raw.strip()

            existing = by_full_club.get((fnk, fcnorm)) or by_full.get(fnk)
            if existing is not None:
                if fnat and not existing["nationality"]:
                    existing["nationality"] = fnat
                if fval and not existing["value"]:
                    existing["value"] = fval
                if fwage and not existing["wage"]:
                    existing["wage"] = fwage
                if fage and not existing["age"]:
                    existing["age"] = fage
                if fpos and not existing["position"]:
                    existing["position"] = fpos.split()[0]
                continue

            if idx < 4500 or fovr >= 74:
                rec_id = f"fifa:{idx}"
                rec = {
                    "id": rec_id,
                    "name": fnm,
                    "full_name": fnm,
                    "common_name": "",
                    "club": fclub or fnat,
                    "club_key": fcnorm or _norm_key(fnat),
                    "ovr": fovr,
                    "position": fpos.split()[0] if fpos else "",
                    "shirt": 0,
                    "nationality": fnat,
                    "age": fage,
                    "value": fval,
                    "wage": fwage,
                    "apps": 0,
                    "starts": 0,
                    "minutes": 0,
                    "seasons": [],
                    "matches": [],
                    "search_keys": [fnk],
                    "source": "FIFA Classic",
                }
                players.append(rec)
                by_id[rec_id] = rec
                _register_keys(rec)

    by_club: dict[str, list[dict]] = {}
    for p in players:
        ck = p["club_key"]
        if ck:
            by_club.setdefault(ck, []).append(p)
    for ck in by_club:
        by_club[ck].sort(key=lambda r: (r["apps"] > 0, r["ovr"], r["apps"]), reverse=True)

    del opta_list, by_full_club, by_init_sur_club, by_sur_club, by_full, fifa_df
    gc.collect()
    return {"players": players, "by_id": by_id, "by_club": by_club}


def search_players(query: str, limit: int = 8) -> list[dict]:
    """Fast normalized search across 25,000+ modern and classic players."""
    qn = _norm_key(query)
    if not qn:
        return []
    raw_tokens = qn.split()
    q_tokens = [TEAM_SEARCH_ALIASES.get(tok, tok) for tok in raw_tokens]
    idx = get_player_search_index()

    scored: list[tuple[tuple, dict]] = []
    for rec in idx["players"]:
        best_tier = None
        club_k = rec["club_key"]
        for sk in rec["search_keys"]:
            words = sk.split()
            if sk == qn:
                tier = 0
            elif len(raw_tokens) == 1 and qn in words:
                tier = 1
            elif sk.startswith(qn) or any(w.startswith(qn) for w in words):
                tier = 2
            elif qn in sk:
                tier = 3
            elif len(q_tokens) > 1 and all((tok in f"{sk} {club_k}") or (rtok in f"{sk} {club_k}") for tok, rtok in zip(q_tokens, raw_tokens)):
                tier = 4
            else:
                continue
            if best_tier is None or tier < best_tier:
                best_tier = tier
        if best_tier is None:
            continue

        score = (
            best_tier,
            -int(rec["ovr"]),
            -min(int(rec["apps"]), 60),
            len(rec["name"]),
        )
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0])
    return [item[1] for item in scored[:limit]]


def open_team_profile(team_name: str) -> None:
    st.session_state["explore_entity"] = {"type": "team", "name": str(team_name)}
    st.session_state["pending_workspace"] = "ws_explore"
    st.rerun()


def open_player_profile(player_rec: dict) -> None:
    st.session_state["explore_entity"] = {
        "type": "player",
        "id": str(player_rec.get("id") or ""),
        "name": str(player_rec.get("name") or ""),
        "club": str(player_rec.get("club") or ""),
    }
    st.session_state["pending_workspace"] = "ws_explore"
    st.rerun()


def _jump_to_catalog_match(comp_name: str, match_dict: dict) -> None:
    if match_dict.get("is_upcoming"):
        st.session_state["jump_upcoming_match_id"] = str(match_dict.get("upcoming_match_id") or "")
        st.session_state["pending_workspace"] = "ws_prematch"
    else:
        st.session_state.pop("daily_match_center_fixture", None)
        st.session_state["jump_comp"] = comp_name
        st.session_state["jump_match_label"] = match_dict.get("label")
        st.session_state["pending_workspace"] = "ws_match"
    st.rerun()


def _render_team_profile(team_name: str, catalog: dict, ui_lang: str) -> None:
    cat_maps = _build_catalog_maps(catalog)
    t_key = _norm_key(team_name)
    team_rec = cat_maps["teams"].get(t_key)
    if team_rec is None:
        clean_k = _clean_club_key(team_name)
        for tk, rec in cat_maps["teams"].items():
            if _clean_club_key(tk) == clean_k:
                team_rec = rec
                break

    disp_name = team_rec["name"] if team_rec else team_name
    matches_all = list(team_rec["matches"]) if team_rec else []
    played_list = sorted(
        [(c, m) for c, m in matches_all if not m.get("is_upcoming")],
        key=lambda x: str(x[1].get("date") or ""),
        reverse=True,
    )
    upcoming_list = sorted(
        [(c, m) for c, m in matches_all if m.get("is_upcoming")],
        key=lambda x: str(x[1].get("date") or "9999-99-99"),
    )
    comps = sorted(team_rec["comps"]) if team_rec else []

    hist_df = load_historical_xg_df()
    form = extract_team_recent_form(hist_df, disp_name, "2099-01-01", n=15) if not hist_df.empty else {}
    elo_val = round(float(form.get("latest_elo", 1500.0))) if form else 0
    avg_xg = float(form.get("xg_pg", 0.0)) if form else 0.0
    avg_xga = float(form.get("xga_pg", 0.0)) if form else 0.0
    wdl_recent = list(form.get("last5_wdl") or []) if form else []

    wdl_html = ""
    for res in wdl_recent:
        bg = "#10B981" if res == "W" else ("#F59E0B" if res == "D" else "#F43F5E")
        wdl_html += (
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:22px;height:22px;border-radius:6px;background:{bg};color:#090D16;'
            f'font-size:11px;font-weight:900;">{res}</span>'
        )
    comp_badges = "".join(
        f'<span style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.28);'
        f'color:#7DD3FC;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;">'
        f'{html.escape(c.split(" ", 1)[-1] if " " in c else c)}</span>'
        for c in comps[:4]
    )

    crest_big = _team_crest_img_html(disp_name, size=62)
    elo_str = f"{elo_val:,}" if elo_val else "N/A"
    xg_str = f"{avg_xg:.2f} / {avg_xga:.2f}" if avg_xg > 0 else "N/A"
    form_label = "Phong độ gần nhất:" if ui_lang == "vi" else "Recent form:"

    st.markdown(compact_html(f"""
    <div style="background:linear-gradient(135deg,#111827 0%,#0B1220 100%);border:1px solid rgba(56,189,248,0.28);border-radius:16px;padding:18px 20px;margin:8px 0 14px 0;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:14px;">
                <div style="width:74px;height:74px;border-radius:14px;background:rgba(15,23,42,0.85);border:1px solid rgba(148,163,184,0.18);display:flex;align-items:center;justify-content:center;">
                    {crest_big}
                </div>
                <div>
                    <div style="font-size:22px;font-weight:900;color:#F8FAFC;line-height:1.15;">{html.escape(disp_name)}</div>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">{comp_badges}</div>
                    {f'<div style="display:flex;align-items:center;gap:6px;margin-top:8px;"><span style="font-size:11px;color:#94A3B8;font-weight:700;">{form_label}</span>{wdl_html}</div>' if wdl_html else ''}
                </div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <div style="background:#0F172A;border:1px solid rgba(148,163,184,0.16);border-radius:10px;padding:8px 14px;text-align:center;min-width:86px;">
                    <div style="font-size:17px;font-weight:900;color:#F8FAFC;">{len(played_list)}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Đã đá</div>
                </div>
                <div style="background:#0F172A;border:1px solid rgba(56,189,248,0.28);border-radius:10px;padding:8px 14px;text-align:center;min-width:86px;">
                    <div style="font-size:17px;font-weight:900;color:#38BDF8;">{len(upcoming_list)}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Sắp đá</div>
                </div>
                <div style="background:#0F172A;border:1px solid rgba(148,163,184,0.16);border-radius:10px;padding:8px 14px;text-align:center;min-width:86px;">
                    <div style="font-size:17px;font-weight:900;color:#FACC15;">{elo_str}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Opta Elo</div>
                </div>
                <div style="background:#0F172A;border:1px solid rgba(16,185,129,0.28);border-radius:10px;padding:8px 14px;text-align:center;min-width:110px;">
                    <div style="font-size:17px;font-weight:900;color:#34D399;">{xg_str}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">xG / xGA TB</div>
                </div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    tab_fx, tab_sq, tab_form = st.tabs(["📅 Lịch Đấu & Kết Quả", "👥 Đội Hình & Cầu Thủ", "📈 Phong Độ & xG"])

    with tab_fx:
        up_df = load_upcoming_fixtures_df()
        if upcoming_list:
            st.markdown("##### 🔮 Trận Sắp Diễn Ra (2026/27)")
            for idx_u, (comp_u, m_u) in enumerate(upcoming_list[:10]):
                h_u, a_u = str(m_u.get("home_team") or ""), str(m_u.get("away_team") or "")
                dt_u = str(m_u.get("date") or "")[:10]
                uid_u = str(m_u.get("upcoming_match_id") or "")
                prob_txt = ""
                if not up_df.empty and uid_u:
                    urow = up_df[up_df["match_id"].astype(str) == uid_u]
                    if not urow.empty:
                        r0 = urow.iloc[0]
                        ph = round(float(r0.get("p_home", 0.34)) * 100)
                        pd_ = round(float(r0.get("p_draw", 0.28)) * 100)
                        pa = max(0, 100 - ph - pd_)
                        prob_txt = f"AI {ph}%-{pd_}%-{pa}% · xG {float(r0.get('xg_home', 1.3)):.2f}-{float(r0.get('xg_away', 1.1)):.2f}"
                c1, c2 = st.columns([3.8, 1.2])
                with c1:
                    st.markdown(compact_html(f"""
                    <div style="background:#0E1420;border:1px solid rgba(56,189,248,0.22);border-radius:10px;padding:9px 12px;margin-bottom:4px;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                        <div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#F8FAFC;font-size:13.5px;">
                            <span style="color:#38BDF8;font-size:11.5px;min-width:74px;">{html.escape(dt_u)}</span>
                            {_team_crest_img_html(h_u, 20)} <span>{html.escape(h_u)}</span>
                            <span style="color:#64748B;">vs</span>
                            {_team_crest_img_html(a_u, 20)} <span>{html.escape(a_u)}</span>
                        </div>
                        <span style="font-size:11.5px;color:#7DD3FC;font-weight:700;">{html.escape(prob_txt)}</span>
                    </div>
                    """), unsafe_allow_html=True)
                with c2:
                    if st.button("🔮 Dự đoán AI", key=f"tm_up_{t_key}_{idx_u}_{uid_u}", width="stretch"):
                        _jump_to_catalog_match(comp_u, m_u)

        st.markdown("##### 🏟️ Kết Quả Đã Đá")
        if not played_list:
            st.info("Chưa có trận đã đá trong dữ liệu sự kiện.")
        for idx_p, (comp_p, m_p) in enumerate(played_list[:25]):
            h_p, a_p = str(m_p.get("home_team") or ""), str(m_p.get("away_team") or "")
            hs, as_ = _safe_int(m_p.get("home_score"), 0), _safe_int(m_p.get("away_score"), 0)
            dt_p = str(m_p.get("date") or "")[:10]
            c1, c2 = st.columns([3.8, 1.2])
            with c1:
                st.markdown(compact_html(f"""
                <div style="background:#0E1420;border:1px solid rgba(148,163,184,0.14);border-radius:10px;padding:9px 12px;margin-bottom:4px;display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">
                    <div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#F1F5F9;font-size:13.5px;">
                        <span style="color:#94A3B8;font-size:11.5px;min-width:74px;">{html.escape(dt_p)}</span>
                        {_team_crest_img_html(h_p, 20)} <span>{html.escape(h_p)}</span>
                        <span style="background:#1E293B;color:#F8FAFC;padding:2px 8px;border-radius:6px;font-weight:900;">{hs} - {as_}</span>
                        <span>{html.escape(a_p)}</span> {_team_crest_img_html(a_p, 20)}
                    </div>
                    <span style="font-size:11px;color:#64748B;">{html.escape(str(m_p.get('stage') or '')[:36])}</span>
                </div>
                """), unsafe_allow_html=True)
            with c2:
                if st.button("🏟️ Xem trận", key=f"tm_pl_{t_key}_{idx_p}_{m_p.get('id')}", width="stretch"):
                    _jump_to_catalog_match(comp_p, m_p)

    with tab_sq:
        p_idx = get_player_search_index()
        ck = _clean_club_key(disp_name)
        squad = list(p_idx["by_club"].get(ck) or [])
        if not squad:
            for k_c, plist in p_idx["by_club"].items():
                if ck and (ck in k_c or k_c in ck):
                    squad.extend(plist)
        squad.sort(key=lambda r: (r["apps"] > 0, r["ovr"], r["apps"]), reverse=True)
        if not squad:
            st.info("Chưa có danh sách cầu thủ chi tiết cho đội bóng này trong kho Opta / EA FC.")
        else:
            st.caption(f"Tìm thấy **{len(squad)}** cầu thủ (hiển thị Top 28). Bấm vào bất kỳ cầu thủ nào để mở hồ sơ chi tiết:")
            cols = st.columns(2)
            for i_s, p_rec in enumerate(squad[:28]):
                with cols[i_s % 2]:
                    portrait = _player_portrait_html(p_rec["name"], disp_name, size=42, allow_network=False)
                    meta_bits = []
                    if p_rec.get("shirt"):
                        meta_bits.append(f"#{p_rec['shirt']}")
                    if p_rec.get("position"):
                        meta_bits.append(p_rec["position"])
                    if p_rec.get("ovr"):
                        meta_bits.append(f"OVR {p_rec['ovr']}")
                    if p_rec.get("apps"):
                        meta_bits.append(f"{p_rec['apps']} trận")
                    st.markdown(compact_html(f"""
                    <div style="background:#0E1420;border:1px solid rgba(148,163,184,0.14);border-radius:10px;padding:8px 10px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
                        {portrait}
                        <div style="min-width:0;flex:1;">
                            <div style="font-weight:800;font-size:13.5px;color:#F8FAFC;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(p_rec['name'])}</div>
                            <div style="font-size:11.5px;color:#94A3B8;">{html.escape(' · '.join(meta_bits) or p_rec.get('source', ''))}</div>
                        </div>
                    </div>
                    """), unsafe_allow_html=True)
                    if st.button(f"👤 Hồ sơ {p_rec['name']}", key=f"sq_p_{t_key}_{i_s}_{p_rec['id']}", width="stretch"):
                        open_player_profile(p_rec)

    with tab_form:
        rows_hist = list(form.get("recent_matches") or []) if form else []
        if not rows_hist:
            st.info("Chưa có dữ liệu chuỗi xG lịch sử cho đội bóng này.")
        else:
            df_f = pd.DataFrame(rows_hist).sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_f["date"], y=df_f["gf"], name="Bàn thắng", marker_color="#38BDF8"))
            fig.add_trace(go.Scatter(x=df_f["date"], y=df_f["xg"], name="xG Kỳ vọng", mode="lines+markers", line=dict(color="#10B981", width=2.5)))
            fig.add_trace(go.Scatter(x=df_f["date"], y=df_f["xga"], name="xGA (Bị đe dọa)", mode="lines+markers", line=dict(color="#F43F5E", width=2, dash="dot")))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1420",
                plot_bgcolor="#0E1420",
                height=340,
                margin=dict(l=24, r=18, t=30, b=24),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
            )
            st.plotly_chart(fig, width="stretch")


def _render_player_profile(entity: dict, catalog: dict, ui_lang: str) -> None:
    p_idx = get_player_search_index()
    rec = p_idx["by_id"].get(str(entity.get("id") or ""))
    if rec is None and entity.get("name"):
        found = search_players(str(entity["name"]), limit=1)
        rec = found[0] if found else None
    if rec is None:
        st.warning("Không tìm thấy hồ sơ cầu thủ phù hợp.")
        return

    pname = rec["name"]
    club = rec.get("club") or rec.get("nationality") or "Free Agent"
    fifa_info = lookup_fifa_player(rec.get("full_name") or pname, pname, club)
    portrait_html = _player_portrait_html(rec.get("full_name") or pname, club, size=86, allow_network=True)
    club_crest = _team_crest_img_html(club, size=24) if rec.get("club") else ""

    ovr = _safe_int(rec.get("ovr"), _safe_int((fifa_info or {}).get("Overall"), 75))
    nat = rec.get("nationality") or (fifa_info or {}).get("Nationality") or "—"
    age = _safe_int(rec.get("age"), _safe_int((fifa_info or {}).get("Age"), 0))
    val_s = rec.get("value") or (fifa_info or {}).get("Value") or "—"
    pos_s = rec.get("position") or (str((fifa_info or {}).get("Preferred Positions") or "").split() or ["—"])[0]
    shirt_badge = f'<span style="background:#1E293B;color:#F8FAFC;padding:3px 9px;border-radius:8px;font-weight:800;font-size:12px;">#{rec["shirt"]}</span>' if rec.get("shirt") else ""

    st.markdown(compact_html(f"""
    <div style="background:linear-gradient(135deg,#111827 0%,#0B1220 100%);border:1px solid rgba(56,189,248,0.28);border-radius:16px;padding:18px 20px;margin:8px 0 14px 0;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:16px;">
                {portrait_html}
                <div>
                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                        <span style="font-size:23px;font-weight:900;color:#F8FAFC;line-height:1.15;">{html.escape(pname)}</span>
                        {shirt_badge}
                        <span style="background:#FACC15;color:#090D16;font-weight:900;font-size:12px;padding:3px 9px;border-radius:8px;">OVR {ovr}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;margin-top:6px;color:#CBD5E1;font-size:13.5px;font-weight:700;">
                        {club_crest} <span>{html.escape(club)}</span>
                        <span style="color:#475569;">•</span>
                        <span>{html.escape(str(pos_s))}</span>
                        <span style="color:#475569;">•</span>
                        <span>{html.escape(str(nat))}{f' ({age} tuổi)' if age else ''}</span>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <div style="background:#0F172A;border:1px solid rgba(56,189,248,0.28);border-radius:10px;padding:8px 14px;text-align:center;min-width:86px;">
                    <div style="font-size:17px;font-weight:900;color:#38BDF8;">{rec['apps']}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Trận Opta</div>
                </div>
                <div style="background:#0F172A;border:1px solid rgba(16,185,129,0.28);border-radius:10px;padding:8px 14px;text-align:center;min-width:104px;">
                    <div style="font-size:17px;font-weight:900;color:#34D399;">{rec['starts']} / {rec['minutes']:,}'</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Đá chính / Phút</div>
                </div>
                <div style="background:#0F172A;border:1px solid rgba(148,163,184,0.16);border-radius:10px;padding:8px 14px;text-align:center;min-width:86px;">
                    <div style="font-size:17px;font-weight:900;color:#FACC15;">{html.escape(str(val_s))}</div>
                    <div style="font-size:10.5px;color:#94A3B8;font-weight:600;">Định giá</div>
                </div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Clickable button to open Club Team Profile if present in catalog
    club_hits = search_teams(catalog, rec["club"], limit=1) if rec.get("club") else []
    if club_hits:
        club_canon = club_hits[0]["name"]
        if st.button(f"🏟️ Xem hồ sơ đội {club_canon}", key=f"pl_open_club_{rec['id']}"):
            open_team_profile(club_canon)

    c_left, c_right = st.columns([1.05, 1.35])
    with c_left:
        st.markdown("##### 📊 Chỉ Số & Thống Kê Mùa Giải")
        if fifa_info:
            radar = compute_player_radar_metrics(
                {"name": pname, "display_name": pname, "pos": str(pos_s), "rating": max(6.4, ovr / 10.0 - 1.2)},
                pd.DataFrame(),
                fifa_info,
            )
            axes_c = radar["axes"] + [radar["axes"][0]]
            vals_c = radar["values"] + [radar["values"][0]]
            fig_r = go.Figure(
                go.Scatterpolar(
                    r=vals_c,
                    theta=axes_c,
                    fill="toself",
                    fillcolor="rgba(56,189,248,0.25)",
                    line=dict(color="#38BDF8", width=2.4),
                    name=pname,
                )
            )
            fig_r.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1420",
                polar=dict(bgcolor="#0E1420", radialaxis=dict(visible=True, range=[30, 100])),
                showlegend=False,
                height=300,
                margin=dict(l=38, r=38, t=20, b=20),
            )
            st.plotly_chart(fig_r, width="stretch")

        if rec["seasons"]:
            df_s = pd.DataFrame(rec["seasons"]).rename(
                columns={"season": "Mùa giải", "competition": "Giải đấu", "club": "CLB", "apps": "Trận", "starts": "Đá chính", "minutes": "Phút"}
            )
            st.dataframe(df_s, width="stretch", hide_index=True)
        elif not fifa_info:
            st.info(f"Nguồn dữ liệu: {rec.get('source', 'EA FC 26/27')} · CLB: {club} · OVR {ovr}")

    with c_right:
        st.markdown("##### ⚽ Trận Đấu Của Cầu Thủ")
        by_opta_id = _build_catalog_maps(catalog)["by_opta_id"]
        shown = 0
        for idx_m, mrow in enumerate(rec["matches"]):
            mdate, mid, comp, season, tname, is_st, mins = mrow
            cat_hit = by_opta_id.get(mid)
            if not cat_hit:
                continue
            comp_name, m_dict = cat_hit
            h_m, a_m = str(m_dict.get("home_team") or ""), str(m_dict.get("away_team") or "")
            hs, as_ = _safe_int(m_dict.get("home_score"), 0), _safe_int(m_dict.get("away_score"), 0)
            role_tag = "Đá chính" if is_st else "Dự bị"
            b1, b2 = st.columns([3.5, 1.3])
            with b1:
                st.markdown(compact_html(f"""
                <div style="background:#0E1420;border:1px solid rgba(148,163,184,0.14);border-radius:10px;padding:8px 11px;margin-bottom:4px;">
                    <div style="font-size:13px;font-weight:800;color:#F8FAFC;">{html.escape(h_m)} {hs} - {as_} {html.escape(a_m)}</div>
                    <div style="font-size:11px;color:#94A3B8;margin-top:2px;">{html.escape(mdate)} · {html.escape(comp)} · {role_tag} · {mins}'</div>
                </div>
                """), unsafe_allow_html=True)
            with b2:
                if st.button("🏟️ Xem trận", key=f"pm_go_{rec['id']}_{idx_m}_{mid}", width="stretch"):
                    _jump_to_catalog_match(comp_name, m_dict)
            shown += 1
            if shown >= 12:
                break
        if shown == 0:
            st.info("Chưa có trận đấu chi tiết từng phút trong kho sự kiện Opta gần đây cho cầu thủ này.")

    # Teammates strip
    teammates = [t for t in (p_idx["by_club"].get(rec["club_key"]) or []) if t["id"] != rec["id"]][:6]
    if teammates:
        st.markdown("##### 👥 Đồng Đội Cùng CLB")
        t_cols = st.columns(min(3, len(teammates)))
        for idx_tm, tm in enumerate(teammates):
            with t_cols[idx_tm % len(t_cols)]:
                if st.button(f"👤 {tm['name']} ({tm['ovr']}★)", key=f"tm_mate_{rec['id']}_{idx_tm}_{tm['id']}", width="stretch"):
                    open_player_profile(tm)


def render_entity_explorer_workspace(catalog: dict, ui_lang: str = "vi") -> None:
    """Dedicated SofaScore/FotMob-style Team & Player Search + Profile Explorer."""
    top_l, top_r = st.columns([1.4, 3.6])
    with top_l:
        if st.button("← Quay lại Lịch Đấu Hôm Nay", key="exp_back_today", width="stretch"):
            st.session_state["pending_workspace"] = "ws_today"
            st.rerun()
    with top_r:
        st.markdown(
            '<div style="font-size:19px;font-weight:900;color:#F8FAFC;padding-top:4px;">'
            "🔍 Khám Phá Đội Bóng & Cầu Thủ</div>",
            unsafe_allow_html=True,
        )

    c_mode, c_q = st.columns([1.2, 2.8])
    with c_mode:
        mode = st.radio(
            "Chế độ tìm kiếm",
            ["team", "player"],
            format_func=lambda x: "🏟️ Đội bóng" if x == "team" else "👤 Cầu thủ",
            horizontal=True,
            key="exp_search_mode",
            label_visibility="collapsed",
        )
    with c_q:
        ph = "Nhập tên đội (VD: Real Madrid, Barca, Arsenal, MU)..." if mode == "team" else "Nhập tên cầu thủ (VD: Mbappe, Yamal, Salah, Haaland, Messi)..."
        q_main = st.text_input("Tìm kiếm", placeholder=ph, key="exp_search_q", label_visibility="collapsed")

    if not str(q_main or "").strip():
        st.caption("Gợi ý tìm nhanh:")
        if mode == "team":
            chips = ["Real Madrid", "Barcelona", "Arsenal", "Liverpool", "Manchester City", "AC Milan"]
            chip_cols = st.columns(len(chips))
            for i_c, nm in enumerate(chips):
                with chip_cols[i_c]:
                    if st.button(f"⚽ {nm}", key=f"exp_chip_t_{i_c}", width="stretch"):
                        open_team_profile(nm)
        else:
            chips = ["Kylian Mbappé", "Lamine Yamal", "Mohamed Salah", "Erling Haaland", "Lionel Messi", "Jude Bellingham"]
            chip_cols = st.columns(len(chips))
            for i_c, nm in enumerate(chips):
                with chip_cols[i_c]:
                    if st.button(f"👤 {nm}", key=f"exp_chip_p_{i_c}", width="stretch"):
                        hits = search_players(nm, limit=1)
                        if hits:
                            open_player_profile(hits[0])
    elif len(q_main.strip()) >= 2:
        if mode == "team":
            t_hits = search_teams(catalog, q_main, limit=9)
            if not t_hits:
                st.info("Không tìm thấy đội bóng phù hợp.")
            else:
                cols = st.columns(3)
                for idx_t, th in enumerate(t_hits):
                    with cols[idx_t % 3]:
                        st.markdown(compact_html(f"""
                        <div style="background:#0E1420;border:1px solid rgba(56,189,248,0.22);border-radius:10px;padding:9px 11px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
                            {_team_crest_img_html(th['name'], 30)}
                            <div style="min-width:0;flex:1;">
                                <div style="font-weight:800;font-size:13.5px;color:#F8FAFC;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(th['name'])}</div>
                                <div style="font-size:11px;color:#94A3B8;">{th['played']} đã đá · {th['upcoming']} sắp đá</div>
                            </div>
                        </div>
                        """), unsafe_allow_html=True)
                        if st.button("Xem đội →", key=f"exp_res_t_{idx_t}_{th['key']}", width="stretch"):
                            open_team_profile(th["name"])
        else:
            p_hits = search_players(q_main, limit=9)
            if not p_hits:
                st.info("Không tìm thấy cầu thủ phù hợp.")
            else:
                cols = st.columns(3)
                for idx_p, ph_rec in enumerate(p_hits):
                    with cols[idx_p % 3]:
                        pt = _player_portrait_html(ph_rec["name"], ph_rec.get("club", ""), size=38, allow_network=False)
                        st.markdown(compact_html(f"""
                        <div style="background:#0E1420;border:1px solid rgba(56,189,248,0.22);border-radius:10px;padding:9px 11px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
                            {pt}
                            <div style="min-width:0;flex:1;">
                                <div style="font-weight:800;font-size:13.5px;color:#F8FAFC;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(ph_rec['name'])}</div>
                                <div style="font-size:11px;color:#94A3B8;">{html.escape(ph_rec.get('club') or '—')} · OVR {ph_rec['ovr']}</div>
                            </div>
                        </div>
                        """), unsafe_allow_html=True)
                        if st.button("Xem hồ sơ →", key=f"exp_res_p_{idx_p}_{ph_rec['id']}", width="stretch"):
                            open_player_profile(ph_rec)

    entity = st.session_state.get("explore_entity")
    if isinstance(entity, dict) and entity.get("type") == "team" and entity.get("name"):
        st.markdown("<hr style='margin:10px 0;border-color:rgba(148,163,184,0.12);'>", unsafe_allow_html=True)
        _render_team_profile(entity["name"], catalog, ui_lang)
    elif isinstance(entity, dict) and entity.get("type") == "player":
        st.markdown("<hr style='margin:10px 0;border-color:rgba(148,163,184,0.12);'>", unsafe_allow_html=True)
        _render_player_profile(entity, catalog, ui_lang)
