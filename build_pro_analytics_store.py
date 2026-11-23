"""
Xây dựng kho dữ liệu Bóng đá Hiện Đại 2025/26 – 2026/27 & World Cup 2026 + Kho Nghiên Cứu Dự Đoán Trận Chưa Đá
Tích hợp:
  1. Opta Event-Level Parquet (2025/26 & 2026/27 Top 5 Leagues + UEL + UCL mới nhất + World Cup 2026 trọn bộ 104 trận)
  2. Kho Dự Đoán & Đặc Trưng Chuyên Sâu (1.928 trận chưa diễn ra + 60.012 trận lịch sử có xG, Elo, Panna/RAPM, 110 chỉ số)
  3. Live ESPN Snapshot (Lịch thi đấu thực tế giờ VN GMT+7, BXH 2026/27, Tỷ lệ kèo DraftKings/ESPN, Vua phá lưới & Nhận định)
"""

import json
import os
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
PRO_DIR = os.path.join(SAMPLE_DIR, "pro_analytics")
MODERN_DIR = os.path.join(SAMPLE_DIR, "opta_modern")
TMP_DIR = os.path.join(SAMPLE_DIR, "_tmp_opta_dl")

os.makedirs(PRO_DIR, exist_ok=True)
os.makedirs(MODERN_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

_CLUB_MAP = {
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Arsenal FC": "Arsenal",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United FC": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Aston Villa FC": "Aston Villa",
    "Leicester City FC": "Leicester City",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Nottingham Forest FC": "Nottingham Forest",
    "Crystal Palace FC": "Crystal Palace",
    "Fulham FC": "Fulham",
    "Everton FC": "Everton",
    "Brentford FC": "Brentford",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds United",
    "Sunderland AFC": "Sunderland",
    "Coventry City FC": "Coventry City",
    "Burnley FC": "Burnley",
    "Southampton FC": "Southampton",
    "AFC Bournemouth": "AFC Bournemouth",
    "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Atlético Madrid",
    "Atlético de Madrid": "Atlético Madrid",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Betis Balompié": "Real Betis",
    "Villarreal CF": "Villarreal",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Girona FC": "Girona",
    "Athletic Club": "Athletic Club",
    "CA Osasuna": "Osasuna",
    "RC Celta de Vigo": "Celta Vigo",
    "Celta de Vigo": "Celta Vigo",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "RCD Mallorca": "Mallorca",
    "Deportivo Alavés": "Alavés",
    "Getafe CF": "Getafe",
    "Reial Club Deportiu Espanyol de Barcelona": "Espanyol",
    "RCD Espanyol de Barcelona": "Espanyol",
    "UD Las Palmas": "Las Palmas",
    "Real Valladolid CF": "Real Valladolid",
    "CD Leganés": "Leganés",
    "Elche CF": "Elche",
    "Levante UD": "Levante",
    "RC Deportivo de La Coruña": "Deportivo La Coruña",
    "Deportivo de La Coruña": "Deportivo La Coruña",
    "Málaga CF": "Málaga",
    "FC Bayern München": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "BV Borussia 09 Dortmund": "Borussia Dortmund",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Borussia VfL Mönchengladbach": "Borussia M'gladbach",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "VfB Stuttgart 1893": "VfB Stuttgart",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "VfL Wolfsburg": "Wolfsburg",
    "SV Werder Bremen": "Werder Bremen",
    "1. FSV Mainz 05": "Mainz 05",
    "FC Augsburg": "Augsburg",
    "1. FC Heidenheim 1846": "Heidenheim",
    "FC St. Pauli 1910": "St. Pauli",
    "1. FC Köln": "Köln",
    "Hamburger SV": "Hamburger SV",
    "FC Schalke 04": "Schalke 04",
    "SV 07 Elversberg": "Elversberg",
    "SC Paderborn 07": "Paderborn",
    "FC Internazionale Milano": "Inter Milan",
    "Internazionale": "Inter Milan",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "AS Roma",
    "Roma": "AS Roma",
    "SS Lazio": "Lazio",
    "Atalanta Bergamasca Calcio": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Genoa CFC": "Genoa",
    "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como",
    "Parma Calcio 1913": "Parma",
    "US Lecce": "Lecce",
    "Hellas Verona FC": "Hellas Verona",
    "Empoli FC": "Empoli",
    "Venezia FC": "Venezia",
    "AC Monza": "Monza",
    "US Sassuolo Calcio": "Sassuolo",
    "Frosinone Calcio": "Frosinone",
    "Paris Saint-Germain FC": "PSG",
    "Paris Saint-Germain": "PSG",
    "Olympique de Marseille": "Marseille",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "Racing Club de Lens": "RC Lens",
    "Lens": "RC Lens",
    "OGC Nice Côte d'Azur": "Nice",
    "OGC Nice": "Nice",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC": "Rennes",
    "RC Strasbourg Alsace": "Strasbourg",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "Stade de Reims": "Reims",
    "Montpellier HSC": "Montpellier",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "AS Saint-Étienne": "Saint-Étienne",
    "FC Lorient": "Lorient",
    "Espérance Sportive Troyes Aube Champagne": "Troyes",
    "SL Benfica": "Benfica",
    "Sporting Clube de Portugal": "Sporting CP",
    "FC Porto": "Porto",
    "Sporting Braga": "Braga",
    "AFC Ajax": "Ajax",
    "PSV Eindhoven": "PSV",
    "Feyenoord Rotterdam": "Feyenoord",
    "Alkmaar Zaanstreek": "AZ Alkmaar",
    "Club Brugge KV": "Club Brugge",
    "Royale Union Saint-Gilloise": "Union SG",
    "Galatasaray Spor Kulübü": "Galatasaray",
    "Galatasaray SK": "Galatasaray",
    "Fenerbahçe Spor Kulübü": "Fenerbahçe",
    "Fenerbahçe SK": "Fenerbahçe",
    "Beşiktaş Jimnastik Kulübü": "Beşiktaş",
    "FC Red Bull Salzburg": "RB Salzburg",
    "BSC Young Boys": "Young Boys",
    "FC Shakhtar Donetsk": "Shakhtar Donetsk",
    "GNK Dinamo Zagreb": "Dinamo Zagreb",
    "FK Crvena zvezda Beograd": "Red Star Belgrade",
    "FK Bodø / Glimt": "Bodø/Glimt",
    "Bodø / Glimt": "Bodø/Glimt",
    "FC København": "Copenhagen",
    "Celtic FC": "Celtic",
    "Rangers FC": "Rangers",
    "Heart of Midlothian FC": "Hearts",
}


def clean_club_name(name: str) -> str:
    s = str(name or "").strip()
    if s in _CLUB_MAP:
        return _CLUB_MAP[s]
    s2 = re.sub(r"^(FC|FK|SK|NK|AC|SC|PFK|HŠK|GNK|KP|KKS|KS|BSC|US|SS|AS|ACF|SSC|RCD|CA|CD|UD|SV|VfL|VfB|TSG|OGC|AJ)\s+", "", s)
    s2 = re.sub(r"\s+(FC|CF|KV|SK|FK|TC|SA|UBK|SCO|AFC|CFC|HSC|1907|1909|1910|1913|1893|1899|1846|1945)$", "", s2)
    s2 = s2.strip() or s
    return _CLUB_MAP.get(s2, s2)


def download_file(url: str, dest: str, timeout: int = 240):
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    return dest


def fetch_json(url: str, timeout: int = 18):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ==========================================================
# PART 1: PRE-MATCH RESEARCH & UPCOMING FIXTURES STORE
# ==========================================================
def build_pro_research_store():
    print("\n=== [1/3] XÂY DỰNG KHO NGHIÊN CỨU & DỰ ĐOÁN TRẬN CHƯA ĐÁ (1.928 FIXTURES + 60K LỊCH SỬ xG) ===")
    pred_raw = os.path.join(PRO_DIR, "predictions.parquet")
    mf_raw = os.path.join(PRO_DIR, "match_features.parquet")
    fx_raw = os.path.join(PRO_DIR, "opta_fixtures.parquet")

    base_rel_pred = "https://github.com/peteowen1/pannadata/releases/download/predictions-latest"
    base_rel_opta = "https://github.com/peteowen1/pannadata/releases/download/opta-latest"

    download_file(f"{base_rel_pred}/predictions.parquet", pred_raw)
    download_file(f"{base_rel_pred}/match_features.parquet", mf_raw)
    download_file(f"{base_rel_opta}/opta_fixtures.parquet", fx_raw)

    pred = pd.read_parquet(pred_raw)
    mf = pd.read_parquet(mf_raw)
    fx = pd.read_parquet(fx_raw, columns=["match_id", "home_score_ht", "away_score_ht", "home_team_id", "away_team_id"])
    fx = fx.drop_duplicates(subset=["match_id"])

    # Normalize team names consistently
    for df_x in (pred, mf):
        df_x["home_team"] = df_x["home_team"].apply(clean_club_name)
        df_x["away_team"] = df_x["away_team"].apply(clean_club_name)
        df_x["match_date"] = df_x["match_date"].astype(str).str.slice(0, 10)

    # 1A. Compact historical_matches_xg.parquet (60,012 played matches with real xG, Elo, HT scores)
    hist_cols = [
        "match_id", "match_date", "league", "season", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg", "home_elo", "away_elo",
        "home_sum_panna", "away_sum_panna",
    ]
    hist = mf[mf["match_status"] == "Played"][[c for c in hist_cols if c in mf.columns]].copy()
    hist = hist.merge(fx[["match_id", "home_score_ht", "away_score_ht"]], on="match_id", how="left")
    hist = hist.sort_values("match_date", ascending=False).reset_index(drop=True)
    hist_out = os.path.join(PRO_DIR, "historical_matches_xg.parquet")
    hist.to_parquet(hist_out, index=False)
    print(f"  -> Đã lưu historical_matches_xg.parquet: {len(hist):,} trận ({os.path.getsize(hist_out)/(1024*1024):.2f} MB)")

    # 1B. Upcoming fixtures merged with predictions + all tactical/squad/Elo features
    up_pred = pred[pred["status"] == "fixture"][
        ["match_id", "match_date", "league", "season", "home_team", "away_team",
         "pred_home_goals", "pred_away_goals", "prob_H", "prob_D", "prob_A",
         "predicted_result", "generated_utc"]
    ].drop_duplicates(subset=["match_id"])

    up_mf = mf[mf["match_status"] == "Fixture"].drop_duplicates(subset=["match_id"])
    extra_cols = [c for c in up_mf.columns if c not in up_pred.columns]
    up_merged = up_pred.merge(up_mf[["match_id"] + extra_cols], on="match_id", how="left")
    up_merged = up_merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    up_out = os.path.join(PRO_DIR, "upcoming_fixtures.parquet")
    up_merged.to_parquet(up_out, index=False)
    print(f"  -> Đã lưu upcoming_fixtures.parquet: {len(up_merged):,} trận chưa diễn ra ({os.path.getsize(up_out)/(1024*1024):.2f} MB)")

    # Remove the big raw 41.8MB match_features.parquet to keep folder compact
    if os.path.isfile(mf_raw):
        os.remove(mf_raw)
    if os.path.isfile(pred_raw):
        os.remove(pred_raw)

    return fx_raw


# ==========================================================
# PART 2: MODERN OPTA EVENT-LEVEL PARQUET STORE (2025/26, 2026/27 & WC 2026)
# ==========================================================
LEAGUE_SPECS = [
    # (opta_comp, asset_name, out_prefix, seasons_keep, category, stage_prefix, arena_default)
    ("World_Cup", "events_World_Cup.parquet", "wc2026", {"2026 Canada-Mexico-USA"}, "🏆 FIFA World Cup", "World Cup 2026", "FIFA World Cup 2026 Stadium"),
    ("EPL", "events_EPL.parquet", "epl", {"2025-2026", "2026-2027"}, "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League", "Premier League", "Premier League Stadium"),
    ("La_Liga", "events_La_Liga.parquet", "laliga", {"2025-2026", "2026-2027"}, "🇪🇸 Spanish La Liga", "La Liga", "La Liga Stadium"),
    ("Serie_A", "events_Serie_A.parquet", "seriea", {"2025-2026", "2026-2027"}, "🇮🇹 Italian Serie A", "Serie A", "Serie A Stadium"),
    ("Bundesliga", "events_Bundesliga.parquet", "bundesliga", {"2025-2026", "2026-2027"}, "🇩🇪 German Bundesliga", "Bundesliga", "Bundesliga Arena"),
    ("Ligue_1", "events_Ligue_1.parquet", "ligue1", {"2025-2026", "2026-2027"}, "🇫🇷 French Ligue 1", "Ligue 1", "Ligue 1 Stadium"),
    ("UEL", "events_UEL.parquet", "uel", {"2025-2026", "2026-2027"}, "🏆 UEFA Europa League", "UEFA Europa League", "UEFA Europa League Arena"),
    ("UCL", "events_UCL.parquet", "ucl_2627", {"2026-2027"}, "⚽ UEFA Champions League", "UEFA Champions League", "UEFA Champions League Arena"),
]


def _infer_stage_name(comp: str, season: str, mdate: str) -> str:
    s_disp = season.replace("-", "/")
    if comp == "World_Cup" and "2026" in season:
        if mdate >= "2026-07-19":
            return "World Cup 2026 - Chung Kết"
        if mdate >= "2026-07-18":
            return "World Cup 2026 - Tranh Hạng 3"
        if mdate >= "2026-07-14":
            return "World Cup 2026 - Bán Kết"
        if mdate >= "2026-07-09":
            return "World Cup 2026 - Tứ Kết"
        if mdate >= "2026-07-04":
            return "World Cup 2026 - Vòng 1/8"
        if mdate >= "2026-06-28":
            return "World Cup 2026 - Vòng 1/16"
        return "World Cup 2026 - Vòng Bảng"
    if comp in ("UCL", "UEL"):
        m = int(mdate[5:7]) if len(mdate) >= 7 and mdate[5:7].isdigit() else 10
        sub = "League Phase" if m in (9, 10, 11, 12, 1) else ("Knockout Play-off / R16" if m in (2, 3) else ("Tứ Kết / Bán Kết" if m == 4 else ("Chung Kết / Bán Kết" if m in (5, 6) else "Vòng Loại")))
        pref = "UEFA Champions League" if comp == "UCL" else "UEFA Europa League"
        return f"{pref} {s_disp} - {sub}"
    prefix_map = {
        "EPL": "Premier League",
        "La_Liga": "La Liga",
        "Serie_A": "Serie A",
        "Bundesliga": "Bundesliga",
        "Ligue_1": "Ligue 1",
    }
    return f"{prefix_map.get(comp, comp)} {s_disp} - Vòng đấu"


def build_modern_event_store(fx_raw_path: str):
    print("\n=== [2/3] XÂY DỰNG KHO SỰ KIỆN OPTA 2025/26 – 2026/27 & WORLD CUP 2026 ===")
    base_rel_opta = "https://github.com/peteowen1/pannadata/releases/download/opta-latest"

    # Load existing UCL match IDs so we only add NEW UCL 2026/27 matches without duplication
    existing_ucl_ids = set()
    ucl_cat_p = os.path.join(SAMPLE_DIR, "opta_ucl", "ucl_catalog.json")
    if os.path.isfile(ucl_cat_p):
        with open(ucl_cat_p, "r", encoding="utf-8") as f:
            for item in json.load(f).get("matches", []):
                if item.get("opta_match_id"):
                    existing_ucl_ids.add(str(item["opta_match_id"]))

    fx_all = pd.read_parquet(fx_raw_path)

    # Download opta_lineups.parquet once
    lu_raw = os.path.join(TMP_DIR, "opta_lineups.parquet")
    print("  -> Đang tải opta_lineups.parquet (81.5 MB)...")
    download_file(f"{base_rel_opta}/opta_lineups.parquet", lu_raw)

    all_catalog_items = []
    all_valid_mids = set()

    def _process_one_league(spec):
        comp, asset_name, out_prefix, seasons_keep, category, _st_pref, arena_default = spec
        out_ev_path = os.path.join(MODERN_DIR, f"{out_prefix}_events.parquet")
        rel_ev_path = f"opta_modern/{out_prefix}_events.parquet"
        rel_lu_path = "opta_modern/modern_lineups.parquet"

        # Check if WC file was already downloaded into PRO_DIR
        local_src = os.path.join(PRO_DIR, asset_name) if os.path.isfile(os.path.join(PRO_DIR, asset_name)) else os.path.join(TMP_DIR, asset_name)
        print(f"  -> [{out_prefix.upper()}] Tải {asset_name}...")
        download_file(f"{base_rel_opta}/{asset_name}", local_src)

        ev_full = pd.read_parquet(local_src)
        ev_sub = ev_full[ev_full["season"].isin(seasons_keep)].copy()
        del ev_full

        fx_sub = fx_all[
            (fx_all["competition"] == comp)
            & (fx_all["season"].isin(seasons_keep))
            & (fx_all["match_status"] == "Played")
        ].drop_duplicates(subset=["match_id"]).copy()

        if comp == "UCL" and existing_ucl_ids:
            fx_sub = fx_sub[~fx_sub["match_id"].astype(str).isin(existing_ucl_ids)].copy()

        valid_mids = set(ev_sub["match_id"].dropna().astype(str).unique())
        fx_sub = fx_sub[fx_sub["match_id"].astype(str).isin(valid_mids)].copy()
        keep_mids = set(fx_sub["match_id"].astype(str).unique())

        ev_sub = ev_sub[ev_sub["match_id"].astype(str).isin(keep_mids)].copy()
        ev_sub = ev_sub.sort_values(["match_id", "period_id", "minute", "second"]).reset_index(drop=True)
        ev_sub.to_parquet(out_ev_path, index=False)

        # Extract goals & penalty shootouts from type_id == 16
        goal_evs = ev_sub[ev_sub["type_id"] == 16][
            ["match_id", "minute", "player_name", "team_id", "period_id", "qualifier_json"]
        ]
        goals_by_mid = {str(mid): grp.to_dict(orient="records") for mid, grp in goal_evs.groupby("match_id")}

        items = []
        for _, r in fx_sub.iterrows():
            mid = str(r["match_id"])
            ht = clean_club_name(r["home_team"] or "Home")
            at = clean_club_name(r["away_team"] or "Away")
            ht_id = str(r["home_team_id"] or "")
            at_id = str(r["away_team_id"] or "")
            hs = int(pd.to_numeric(r["home_score"], errors="coerce") or 0)
            as_ = int(pd.to_numeric(r["away_score"], errors="coerce") or 0)
            season_s = str(r.get("season") or "")
            mdate = str(r.get("match_date") or "")[:10]

            goals_list = []
            pen_h, pen_a, has_pen = 0, 0, False
            for gr in goals_by_mid.get(mid, []):
                pid = int(gr.get("period_id") or 1)
                q_str = str(gr.get("qualifier_json") or "")
                tid = str(gr.get("team_id") or "")
                is_og = ('"OWN_GOAL"' in q_str) or ('"28":' in q_str)
                if pid == 5:
                    has_pen = True
                    if tid == ht_id:
                        pen_h += 1
                    elif tid == at_id:
                        pen_a += 1
                    continue
                if pid not in (1, 2, 3, 4):
                    continue
                ben_team = (at if tid == ht_id else ht) if is_og else (ht if tid == ht_id else at)
                m_disp = int(gr.get("minute") or 0) + 1
                goals_list.append({
                    "player": str(gr.get("player_name") or "Cầu thủ").strip() or "Cầu thủ",
                    "minute": m_disp,
                    "team": ben_team,
                    "is_og": is_og,
                })
            goals_list.sort(key=lambda x: x["minute"])
            g_home_cnt = sum(1 for g in goals_list if g.get("team") == ht)
            g_away_cnt = sum(1 for g in goals_list if g.get("team") == at)
            hs = max(hs, g_home_cnt)
            as_ = max(as_, g_away_cnt)
            pen_str = f" (Pen {pen_h}-{pen_a})" if has_pen else ""
            stage_label = _infer_stage_name(comp, season_s, mdate)
            star_sfx = " ★" if ("Chung Kết" in stage_label and comp == "World_Cup") else ""

            items.append({
                "category": category,
                "id": f"opta_mod_{out_prefix}_{mid}",
                "opta_match_id": mid,
                "label": f"{ht} {hs} - {as_} {at}{pen_str}{star_sfx}",
                "home_team": ht,
                "away_team": at,
                "home_team_id": ht_id,
                "away_team_id": at_id,
                "home_score": hs,
                "away_score": as_,
                "pen_home": pen_h if has_pen else None,
                "pen_away": pen_a if has_pen else None,
                "stage": stage_label,
                "date": mdate,
                "stadium": arena_default,
                "referee": "FIFA / UEFA Official Referee",
                "goals": goals_list,
                "source_type": "opta_modern_parquet",
                "events_rel_path": rel_ev_path,
                "lineups_rel_path": rel_lu_path,
            })

        # Clean up raw downloaded file immediately to save disk space
        if os.path.isfile(local_src):
            try:
                os.remove(local_src)
            except OSError:
                pass

        sz_mb = os.path.getsize(out_ev_path) / (1024 * 1024)
        print(f"  ✅ [{out_prefix.upper()}] Hoàn tất: {len(items)} trận ({len(ev_sub):,} sự kiện, {sz_mb:.1f} MB)")
        return items, keep_mids

    # Run downloads & filtering with 3 parallel workers
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_process_one_league, sp) for sp in LEAGUE_SPECS]
        for fut in as_completed(futs):
            its, mids = fut.result()
            all_catalog_items.extend(its)
            all_valid_mids.update(mids)

    print(f"  -> Đang trích xuất modern_lineups.parquet cho {len(all_valid_mids):,} trận...")
    lu_df = pd.read_parquet(lu_raw)
    lu_mod = lu_df[lu_df["match_id"].astype(str).isin(all_valid_mids)].sort_values(["match_id"]).reset_index(drop=True)
    lu_out = os.path.join(MODERN_DIR, "modern_lineups.parquet")
    lu_mod.to_parquet(lu_out, index=False)
    print(f"  ✅ Đã lưu modern_lineups.parquet ({len(lu_mod):,} dòng, {os.path.getsize(lu_out)/(1024*1024):.1f} MB)")

    all_catalog_items.sort(key=lambda x: x.get("date") or "", reverse=True)
    cat_out = os.path.join(MODERN_DIR, "modern_catalog.json")
    with open(cat_out, "w", encoding="utf-8") as f:
        json.dump({"count": len(all_catalog_items), "matches": all_catalog_items}, f, ensure_ascii=False)
    print(f"  ✅ Đã lưu modern_catalog.json ({len(all_catalog_items):,} trận 2025/26 – 2026/27 & World Cup 2026)")

    # Cleanup temp files
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    if os.path.isfile(fx_raw_path):
        try:
            os.remove(fx_raw_path)
        except OSError:
            pass


# ==========================================================
# PART 3: LIVE ESPN SNAPSHOT (ODDS, KICKOFF GMT+7, STANDINGS, LEADERS)
# ==========================================================
ESPN_LEAGUES = [
    ("ENG", "eng.1", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League"),
    ("ESP", "esp.1", "🇪🇸 Spanish La Liga"),
    ("ITA", "ita.1", "🇮🇹 Italian Serie A"),
    ("GER", "ger.1", "🇩🇪 German Bundesliga"),
    ("FRA", "fra.1", "🇫🇷 French Ligue 1"),
    ("UCL", "uefa.champions", "⚽ UEFA Champions League"),
    ("UEL", "uefa.europa", "🏆 UEFA Europa League"),
]


def _american_to_implied_prob(ml):
    try:
        v = float(ml)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    if v < 0:
        return (-v) / (-v + 100.0)
    return 100.0 / (v + 100.0)


def build_espn_live_cache():
    print("\n=== [3/3] ĐỒNG BỘ DỮ LIỆU LIVE ESPN (KÈO NHÀ CÁI, GIỜ THI ĐẤU GMT+7, BXH 2026/27) ===")
    now_utc = datetime.now(timezone.utc)
    vn_tz = timezone(timedelta(hours=7))
    dates_to_fetch = [(now_utc + timedelta(days=d)).strftime("%Y%m%d") for d in range(-1, 10)]

    live_fixtures = []
    live_standings = {}

    for code, slug, comp_label in ESPN_LEAGUES:
        # 1. Standings
        try:
            st_raw = fetch_json(f"https://site.api.espn.com/apis/v2/sports/soccer/{slug}/standings")
            ch = st_raw.get("children", [])
            rows = []
            if ch:
                for ent in ch[0].get("standings", {}).get("entries", []):
                    t_info = ent.get("team", {})
                    s_map = {s.get("name"): s.get("displayValue") for s in ent.get("stats", [])}
                    rows.append({
                        "rank": int(float(s_map.get("rank") or len(rows) + 1)),
                        "team": clean_club_name(t_info.get("displayName", "")),
                        "team_raw": t_info.get("displayName", ""),
                        "logo": (t_info.get("logos") or [{}])[0].get("href", ""),
                        "mp": int(float(s_map.get("gamesPlayed") or 0)),
                        "w": int(float(s_map.get("wins") or 0)),
                        "d": int(float(s_map.get("ties") or 0)),
                        "l": int(float(s_map.get("losses") or 0)),
                        "gf": int(float(s_map.get("pointsFor") or 0)),
                        "ga": int(float(s_map.get("pointsAgainst") or 0)),
                        "gd": int(float(str(s_map.get("pointDifferential") or "0").replace("+", "") or 0)),
                        "pts": int(float(s_map.get("points") or 0)),
                    })
                rows.sort(key=lambda r: (r["rank"], -r["pts"], -r["gd"]))
            live_standings[code] = {"league_name": comp_label, "slug": slug, "rows": rows}
        except Exception as e:
            print(f"  ⚠️ Standings {slug}: {e}")

        # 2. Scoreboard (yesterday, live today, and next 10 days with GMT+7 times)
        seen_ids = set()
        for dt_s in dates_to_fetch:
            try:
                sb = fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={dt_s}")
                for ev in sb.get("events", []):
                    eid = str(ev.get("id") or "")
                    if not eid or eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    st_obj = ev.get("status") or {}
                    st_type = st_obj.get("type") or {}
                    st_name = str(st_type.get("name") or "")
                    state = str(st_type.get("state") or "pre").lower()
                    if state not in ("in", "pre", "post"):
                        state = "post" if ("FINAL" in st_name or "FULL_TIME" in st_name) else "pre"
                    comp0 = (ev.get("competitions") or [{}])[0]
                    comps = comp0.get("competitors", [])
                    home_c = next((c for c in comps if c.get("homeAway") == "home"), None)
                    away_c = next((c for c in comps if c.get("homeAway") == "away"), None)
                    if not home_c or not away_c:
                        continue
                    h_name = clean_club_name(home_c.get("team", {}).get("displayName", ""))
                    a_name = clean_club_name(away_c.get("team", {}).get("displayName", ""))
                    h_score, a_score = None, None
                    if state in ("in", "post"):
                        try:
                            h_score = int(float(home_c.get("score")))
                            a_score = int(float(away_c.get("score")))
                        except (TypeError, ValueError):
                            h_score, a_score = None, None
                    kickoff_utc = str(ev.get("date") or "")
                    vn_date, vn_time = kickoff_utc[:10], ""
                    if "T" in kickoff_utc:
                        try:
                            dt_u = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
                            dt_v = dt_u.astimezone(vn_tz)
                            vn_date, vn_time = dt_v.strftime("%Y-%m-%d"), dt_v.strftime("%H:%M")
                        except Exception:
                            pass
                    raw_detail = str(st_type.get("shortDetail") or "").strip()
                    if state == "in":
                        status_detail = raw_detail or str(st_obj.get("displayClock") or "LIVE")
                    elif state == "post":
                        status_detail = raw_detail or "FT"
                    else:
                        status_detail = "Hoãn" if "POSTPONED" in st_name.upper() else (vn_time or "Theo lịch")
                    venue = comp0.get("venue", {}).get("fullName", "")
                    city = comp0.get("venue", {}).get("address", {}).get("city", "")
                    odds_arr = comp0.get("odds") or []
                    od0 = next((o for o in odds_arr if isinstance(o, dict)), None)
                    odds_info = {}
                    if od0:
                        h_ml = (od0.get("homeTeamOdds") or {}).get("moneyLine")
                        a_ml = (od0.get("awayTeamOdds") or {}).get("moneyLine")
                        d_ml = (od0.get("drawOdds") or {}).get("moneyLine")
                        ph, pd_, pa = _american_to_implied_prob(h_ml), _american_to_implied_prob(d_ml), _american_to_implied_prob(a_ml)
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
                    live_fixtures.append({
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
                        "home_logo": home_c.get("team", {}).get("logo", ""),
                        "away_logo": away_c.get("team", {}).get("logo", ""),
                        "home_score": h_score,
                        "away_score": a_score,
                        "home_form": home_c.get("form", ""),
                        "away_form": away_c.get("form", ""),
                        "venue": f"{venue} ({city})" if venue and city else venue,
                        "odds": odds_info,
                    })
            except Exception:
                continue

    cache_payload = {
        "updated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "standings": live_standings,
        "live_fixtures": live_fixtures,
    }
    out_p = os.path.join(PRO_DIR, "espn_live_cache.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False)
    print(f"  ✅ Đã lưu espn_live_cache.json: {len(live_fixtures)} trận Live sắp đá + {len(live_standings)} BXH thực tế 2026/27!")


if __name__ == "__main__":
    fx_raw = build_pro_research_store()
    build_modern_event_store(fx_raw)
    build_espn_live_cache()
    print("\n🎉 HOÀN TẤT TOÀN BỘ KHO DỮ LIỆU BÓNG ĐÁ HIỆN ĐẠI 2025/26 – 2026/27 & PRE-MATCH AI!")
