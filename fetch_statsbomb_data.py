"""
Bộ nạp dữ liệu toàn diện từ StatsBomb Open Data (https://github.com/statsbomb/open-data):
- 100% Trọn bộ 64/64 trận FIFA World Cup 2022 + 19 trận Kinh điển Lịch sử World Cup
- 100% Trọn bộ 51/51 trận UEFA Euro 2024 + 51/51 trận UEFA Euro 2020 (102 trận)
- 100% Trọn bộ 32/32 trận CONMEBOL Copa América 2024
- 100% Trọn bộ 52/52 trận Africa Cup of Nations (AFCON 2023)
- 100% Trọn bộ 18/18 trận Chung kết UEFA Champions League trên StatsBomb + 3 trận Europa League
- 100% Trọn bộ 34/34 trận Bayer Leverkusen Bất bại Bundesliga 2023/24 + Đại chiến 2015/16
- 100% Trọn bộ 38/38 trận Arsenal Invincibles 2003/04 + Đại chiến Big-6 Premier League 2015/16
- Trọn bộ La Liga 2020/21, 2019/20 + Toàn bộ Siêu kinh điển El Clásico (2004–2019)
- Trọn bộ Ligue 1 2022/23 & 2021/22 (PSG: Messi - Mbappé - Neymar) + Đại chiến Serie A
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
EVENTS_DIR = os.path.join(SAMPLE_DIR, "events")
LINEUPS_DIR = os.path.join(SAMPLE_DIR, "lineups")
MANIFEST_PATH = os.path.join(SAMPLE_DIR, "statsbomb_catalog.json")

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(LINEUPS_DIR, exist_ok=True)


def _fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "SofaScoreAnalyticsPlatform/2.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_goals_and_pens(raw_events: list, home_team: str, away_team: str):
    goals = []
    pen_home = 0
    pen_away = 0
    has_pen = False
    for e in raw_events:
        period = e.get("period", 1)
        t_name = (e.get("type") or {}).get("name", "")
        team = (e.get("team") or {}).get("name", "")
        if period == 5:
            if t_name == "Shot":
                has_pen = True
                if ((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
                    if team == home_team:
                        pen_home += 1
                    elif team == away_team:
                        pen_away += 1
            continue
        if period not in (1, 2, 3, 4):
            continue
        player = (e.get("player") or {}).get("name", "")
        minute = int(e.get("minute", 0) or 0)
        if t_name == "Shot" and ((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
            goals.append({"player": player, "minute": minute, "team": team, "is_og": False})
        elif t_name == "Own Goal Against":
            ben_team = away_team if team == home_team else home_team
            goals.append({"player": player, "minute": minute, "team": ben_team, "is_og": True})
    goals.sort(key=lambda g: g["minute"])
    return goals, (pen_home if has_pen else None), (pen_away if has_pen else None)


def _select_matches_to_download():
    selected = []
    already_hardcoded = {3869685, 3869519, 3943043, 22912, 303731}

    # 1. FIFA WORLD CUP: 100% trọn bộ 64 trận WC 2022 + toàn bộ các mùa lịch sử (trừ WC 2018 đã nạp từ CSV)
    wc_seasons = [
        (106, "World Cup 2022"),
        (55, "World Cup 1990"),
        (54, "World Cup 1986"),
        (51, "World Cup 1974"),
        (272, "World Cup 1970"),
        (270, "World Cup 1962"),
        (269, "World Cup 1958"),
    ]
    for sid, sprefix in wc_seasons:
        for m in _fetch_json(f"{RAW_BASE}/matches/43/{sid}.json"):
            if int(m["match_id"]) in already_hardcoded:
                continue
            stg = (m.get("competition_stage") or {}).get("name", "Group Stage")
            grp = (m.get("home_team") or {}).get("home_team_group") or ""
            stg_full = f"{sprefix} - {stg}" + (f" ({grp})" if grp and stg == "Group Stage" else "")
            selected.append(("🏆 FIFA World Cup", stg_full, m))

    # 2. UEFA EURO: 100% trọn bộ 51 trận Euro 2024 + 51 trận Euro 2020 (102/102 trận)
    for sid, sprefix in [(282, "Euro 2024"), (43, "Euro 2020")]:
        for m in _fetch_json(f"{RAW_BASE}/matches/55/{sid}.json"):
            if int(m["match_id"]) in already_hardcoded:
                continue
            stg = (m.get("competition_stage") or {}).get("name", "Group Stage")
            grp = (m.get("home_team") or {}).get("home_team_group") or ""
            stg_full = f"{sprefix} - {stg}" + (f" ({grp})" if grp and stg == "Group Stage" else "")
            selected.append(("🏆 UEFA European Championship", stg_full, m))

    # 3. CONMEBOL COPA AMÉRICA 2024: 100% trọn bộ 32/32 trận
    for m in _fetch_json(f"{RAW_BASE}/matches/223/282.json"):
        stg = (m.get("competition_stage") or {}).get("name", "Group Stage")
        grp = (m.get("home_team") or {}).get("home_team_group") or ""
        stg_full = f"Copa América 2024 - {stg}" + (f" ({grp})" if grp and stg == "Group Stage" else "")
        selected.append(("🏆 CONMEBOL Copa América", stg_full, m))

    # 4. AFRICA CUP OF NATIONS 2023: 100% trọn bộ 52/52 trận
    for m in _fetch_json(f"{RAW_BASE}/matches/1267/107.json"):
        stg = (m.get("competition_stage") or {}).get("name", "Group Stage")
        grp = (m.get("home_team") or {}).get("home_team_group") or ""
        stg_full = f"AFCON 2023 - {stg}" + (f" ({grp})" if grp and stg == "Group Stage" else "")
        selected.append(("🌍 Africa Cup of Nations (AFCON)", stg_full, m))

    # 5. UEFA CHAMPIONS LEAGUE: 100% toàn bộ 18 mùa trên StatsBomb
    comps = _fetch_json(f"{RAW_BASE}/competitions.json")
    for c in comps:
        if c["competition_id"] == 16:
            sid, sname = c["season_id"], c["season_name"]
            for m in _fetch_json(f"{RAW_BASE}/matches/16/{sid}.json"):
                if int(m["match_id"]) not in already_hardcoded:
                    stg = (m.get("competition_stage") or {}).get("name", "Final")
                    selected.append(("⚽ UEFA Champions League", f"Champions League {sname} - {stg}", m))

    # 6. ENGLISH PREMIER LEAGUE: Trọn bộ 38 trận Arsenal Invincibles 2003/04 + Đại chiến Big-6 & Leicester 2015/16
    for m in _fetch_json(f"{RAW_BASE}/matches/2/44.json"):
        mw = m.get("match_week", "")
        selected.append(("🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League", f"Premier League 2003/04 - Vòng {mw}", m))

    big6_epl = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham Hotspur", "Leicester City"}
    for m in _fetch_json(f"{RAW_BASE}/matches/2/27.json"):
        ht, at = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        if ht in big6_epl and at in big6_epl:
            mw = m.get("match_week", "")
            selected.append(("🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League", f"Premier League 2015/16 - Vòng {mw}", m))

    # 7. SPANISH LA LIGA: Trọn bộ 2020/21 + 2019/20 + Toàn bộ El Clásico lịch sử
    for sid, sname in [(90, "2020/21"), (42, "2019/20")]:
        for m in _fetch_json(f"{RAW_BASE}/matches/11/{sid}.json"):
            if int(m["match_id"]) not in already_hardcoded:
                mw = m.get("match_week", "")
                selected.append(("🇪🇸 Spanish La Liga", f"La Liga {sname} - Vòng {mw}", m))
    for c in comps:
        if c["competition_id"] == 11 and c["season_id"] not in (90, 42):
            sid, sname = c["season_id"], c["season_name"]
            for m in _fetch_json(f"{RAW_BASE}/matches/11/{sid}.json"):
                ht, at = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
                if frozenset({ht, at}) == frozenset({"Barcelona", "Real Madrid"}):
                    selected.append(("🇪🇸 Spanish La Liga", f"La Liga {sname} - El Clásico", m))

    # 8. GERMAN BUNDESLIGA: 100% trọn bộ 34/34 trận Leverkusen Bất bại 2023/24 + Top trận 2015/16
    for m in _fetch_json(f"{RAW_BASE}/matches/9/281.json"):
        mw = m.get("match_week", "")
        selected.append(("🇩🇪 German Bundesliga", f"Bundesliga 2023/24 - Vòng {mw}", m))
    top_ger = {"Bayern Munich", "Borussia Dortmund", "Bayer Leverkusen", "RB Leipzig", "Schalke 04", "Borussia Mönchengladbach"}
    for m in _fetch_json(f"{RAW_BASE}/matches/9/27.json"):
        ht, at = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        if ht in top_ger and at in top_ger:
            mw = m.get("match_week", "")
            selected.append(("🇩🇪 German Bundesliga", f"Bundesliga 2015/16 - Vòng {mw}", m))

    # 9. FRENCH LIGUE 1 (58 trận PSG Messi-Mbappe-Neymar 2021-2023) & ITALIAN SERIE A (Đại chiến Top 6)
    for sid, sname in [(235, "2022/23"), (108, "2021/22")]:
        for m in _fetch_json(f"{RAW_BASE}/matches/7/{sid}.json"):
            mw = m.get("match_week", "")
            selected.append(("🇫🇷 French Ligue 1", f"Ligue 1 {sname} - Vòng {mw}", m))

    top_ita = {"Juventus", "AC Milan", "Inter", "Napoli", "AS Roma", "Lazio", "Fiorentina"}
    for m in _fetch_json(f"{RAW_BASE}/matches/12/27.json"):
        ht, at = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        if ht in top_ita and at in top_ita:
            mw = m.get("match_week", "")
            selected.append(("🇮🇹 Italian Serie A", f"Serie A 2015/16 - Vòng {mw}", m))

    uniq = {}
    for cat, stg_lbl, m in selected:
        uniq[int(m["match_id"])] = (cat, stg_lbl, m)
    return list(uniq.values())


def _download_single_match(item):
    cat, stg_lbl, m = item
    mid = int(m["match_id"])
    ev_path = os.path.join(EVENTS_DIR, f"{mid}.json")
    lu_path = os.path.join(LINEUPS_DIR, f"{mid}.json")

    if not os.path.isfile(ev_path) or os.path.getsize(ev_path) < 500:
        ev_data = _fetch_json(f"{RAW_BASE}/events/{mid}.json", timeout=40)
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(ev_data, f, ensure_ascii=False)
    else:
        with open(ev_path, "r", encoding="utf-8") as f:
            ev_data = json.load(f)

    if not os.path.isfile(lu_path) or os.path.getsize(lu_path) < 200:
        try:
            lu_data = _fetch_json(f"{RAW_BASE}/lineups/{mid}.json", timeout=25)
            with open(lu_path, "w", encoding="utf-8") as f:
                json.dump(lu_data, f, ensure_ascii=False)
        except Exception:
            pass

    ht = m["home_team"]["home_team_name"]
    at = m["away_team"]["away_team_name"]
    hs = int(m.get("home_score", 0))
    as_ = int(m.get("away_score", 0))
    goals, pen_h, pen_a = _extract_goals_and_pens(ev_data, ht, at)
    pen_str = f" (Pen {pen_h}-{pen_a})" if pen_h is not None else ""
    star = " ★" if ("Final" in stg_lbl and "3rd" not in stg_lbl and "Quarter" not in stg_lbl and "Semi" not in stg_lbl) or "El Clásico" in stg_lbl else ""
    stadium = (m.get("stadium") or {}).get("name") or "International Stadium"
    referee = (m.get("referee") or {}).get("name") or "Official Referee"
    grp = (m.get("home_team") or {}).get("home_team_group") or ""

    return {
        "category": cat,
        "id": f"sb_{mid}",
        "statsbomb_match_id": str(mid),
        "label": f"{ht} {hs} - {as_} {at}{pen_str}{star}",
        "home_team": ht,
        "away_team": at,
        "home_score": hs,
        "away_score": as_,
        "pen_home": pen_h,
        "pen_away": pen_a,
        "group": grp,
        "stage": stg_lbl,
        "date": str(m.get("match_date") or "")[:10],
        "stadium": stadium,
        "referee": referee,
        "goals": goals,
        "source_type": "json_file",
        "rel_path": f"events/{mid}.json",
    }


def main():
    print("🔍 Đang quét toàn bộ kho StatsBomb Open Data...")
    targets = _select_matches_to_download()
    print(f"📦 Tổng số trận đấu cần đồng bộ: {len(targets)} trận...")

    manifest_items = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_download_single_match, it): it for it in targets}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                rec = fut.result()
                manifest_items.append(rec)
                done += 1
                if done % 40 == 0 or done == len(targets):
                    print(f"   ✅ Tiến độ: {done}/{len(targets)} trận (vừa xong: {rec['label']})")
            except Exception as exc:
                print(f"   ⚠️ Lỗi match {it[2].get('match_id')}: {exc}")

    # Giữ lại các trận Opta UCL (nếu đã nạp trước đó trong manifest)
    opta_existing = []
    if os.path.isfile(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
                old_m = json.load(mf).get("matches", [])
            opta_existing = [x for x in old_m if str(x.get("id", "")).startswith("opta_")]
        except Exception:
            pass

    all_items = manifest_items + opta_existing
    all_items.sort(key=lambda r: (r["category"], r.get("date") or ""), reverse=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": len(all_items), "matches": all_items}, f, ensure_ascii=False, indent=2)

    by_cat = {}
    for r in all_items:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("\n🎉 ĐỒNG BỘ HOÀN TẤT!")
    for k, v in sorted(by_cat.items()):
        print(f"   • {k}: {v} trận")


if __name__ == "__main__":
    main()
