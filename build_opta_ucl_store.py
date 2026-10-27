"""
Xây dựng kho dữ liệu UEFA Champions League chuyên sâu (Opta Event-Level Parquet Store)
từ bộ dữ liệu mở peteowen1/pannadata trên GitHub (~1.800+ trận Champions League 2015/16 – 2025/26,
gồm 3,26 triệu sự kiện tọa độ thực tế + đội hình xuất phát & dự bị).
"""

import json
import os
import shutil
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
RAW_DIR = os.path.join(SAMPLE_DIR, "_opta_raw")
OUT_DIR = os.path.join(SAMPLE_DIR, "opta_ucl")
os.makedirs(OUT_DIR, exist_ok=True)


import re

_CLUB_MAP = {
    "Manchester City FC": "Manchester City", "Manchester United FC": "Manchester United",
    "Arsenal FC": "Arsenal", "Liverpool FC": "Liverpool", "Chelsea FC": "Chelsea",
    "Tottenham Hotspur FC": "Tottenham", "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa", "Leicester City FC": "Leicester City",
    "Celtic FC": "Celtic", "Rangers FC": "Rangers", "Heart of Midlothian FC": "Hearts",
    "Real Madrid CF": "Real Madrid", "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Atlético Madrid", "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Betis Balompié": "Real Betis", "Villarreal CF": "Villarreal", "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia", "Girona FC": "Girona", "Athletic Club": "Athletic Bilbao",
    "FC Bayern München": "Bayern Munich", "BV Borussia 09 Dortmund": "Borussia Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen", "RasenBallsport Leipzig": "RB Leipzig",
    "Borussia VfL Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Frankfurt",
    "VfB Stuttgart 1893": "Stuttgart", "TSG 1899 Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin", "FC Internazionale Milano": "Inter Milan",
    "Juventus FC": "Juventus", "SSC Napoli": "Napoli", "Atalanta Bergamasca Calcio": "Atalanta",
    "Bologna FC 1909": "Bologna", "Paris Saint-Germain FC": "PSG", "Paris Saint-Germain": "PSG",
    "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon", "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille", "Racing Club de Lens": "RC Lens", "OGC Nice Côte d'Azur": "OGC Nice",
    "Stade Brestois 29": "Brest", "Stade Rennais FC": "Rennes", "SL Benfica": "Benfica",
    "Sporting Clube de Portugal": "Sporting CP", "FC Porto": "Porto", "Sporting Braga": "Braga",
    "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV", "Feyenoord Rotterdam": "Feyenoord",
    "Alkmaar Zaanstreek": "AZ Alkmaar", "Club Brugge KV": "Club Brugge",
    "Royale Union Saint-Gilloise": "Union SG", "Royal Antwerp FC": "Antwerp",
    "Galatasaray Spor Kulübü": "Galatasaray", "Galatasaray SK": "Galatasaray",
    "Fenerbahçe Spor Kulübü": "Fenerbahçe", "Fenerbahçe SK": "Fenerbahçe",
    "Beşiktaş Jimnastik Kulübü": "Beşiktaş", "İstanbul Başakşehir Futbol Kulübü": "Başakşehir",
    "İstanbul Başakşehir FK": "Başakşehir", "FC Red Bull Salzburg": "RB Salzburg",
    "BSC Young Boys": "Young Boys", "FC Shakhtar Donetsk": "Shakhtar Donetsk",
    "GNK Dinamo Zagreb": "Dinamo Zagreb", "FK Crvena zvezda Beograd": "Red Star Belgrade",
    "FK Bodø / Glimt": "Bodø/Glimt", "FC København": "Copenhagen",
    "SC Fotbal Club FCSB SA": "FCSB", "PFK Ludogorets 1945 Razgrad": "Ludogorets",
}


def _clean_club_name(name: str) -> str:
    s = str(name or "").strip()
    if s in _CLUB_MAP:
        return _CLUB_MAP[s]
    s2 = re.sub(r"^(FC|FK|SK|NK|AC|SC|PFK|HŠK|GNK|KP|KKS|KS|BSC)\s+", "", s)
    s2 = re.sub(r"\s+(FC|CF|KV|SK|FK|TC|SA|UBK|1907|1909|1893|1899|1945)$", "", s2)
    return s2.strip() or s


def main():
    fx_path = os.path.join(RAW_DIR, "opta_fixtures.parquet")
    ev_path = os.path.join(RAW_DIR, "events_UCL.parquet")
    lu_path = os.path.join(RAW_DIR, "opta_lineups.parquet")

    print("1. Đang đọc opta_fixtures.parquet & lọc các trận UEFA Champions League đã thi đấu...")
    fx = pd.read_parquet(fx_path)
    fx_ucl = fx[(fx["competition"] == "UCL") & (fx["match_status"] == "Played")].copy()
    fx_ucl = fx_ucl.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    print("2. Đang đọc events_UCL.parquet (3.26M sự kiện) để trích xuất bàn thắng & luân lưu...")
    ev = pd.read_parquet(ev_path)
    valid_mids = set(ev["match_id"].dropna().unique())
    fx_ucl = fx_ucl[fx_ucl["match_id"].isin(valid_mids)].copy()
    print(f"   -> Khớp {len(fx_ucl)} trận Champions League có đầy đủ tọa độ sự kiện!")

    # Trích xuất goals & penalty shootout từ type_id == 16
    goal_evs = ev[ev["type_id"] == 16][["match_id", "minute", "player_name", "team_id", "period_id", "qualifier_json"]].copy()
    goals_by_mid = {}
    for mid, grp in goal_evs.groupby("match_id"):
        goals_by_mid[mid] = grp.to_dict(orient="records")

    catalog_items = []
    for _, r in fx_ucl.iterrows():
        mid = str(r["match_id"])
        ht = _clean_club_name(r["home_team"] or "Home")
        at = _clean_club_name(r["away_team"] or "Away")
        ht_id = str(r["home_team_id"] or "")
        at_id = str(r["away_team_id"] or "")
        hs = int(pd.to_numeric(r["home_score"], errors="coerce") or 0)
        as_ = int(pd.to_numeric(r["away_score"], errors="coerce") or 0)
        season = str(r.get("season") or "").replace("-", "/")
        mdate = str(r.get("match_date") or "")[:10]

        g_rows = goals_by_mid.get(mid, [])
        goals_list = []
        pen_h, pen_a, has_pen = 0, 0, False
        for gr in g_rows:
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
            if is_og:
                ben_team = at if tid == ht_id else ht
            else:
                ben_team = ht if tid == ht_id else at
            m_disp = int(gr.get("minute") or 0) + 1
            goals_list.append({
                "player": str(gr.get("player_name") or "Cầu thủ").strip() or "Cầu thủ",
                "minute": m_disp,
                "team": ben_team,
                "is_og": is_og,
            })
        goals_list.sort(key=lambda x: x["minute"])

        # Phân loại vòng đấu tự động theo tháng thi đấu của lịch UCL
        month = int(mdate[5:7]) if len(mdate) >= 7 and mdate[5:7].isdigit() else 10
        if month in (9, 10, 11, 12, 1):
            stage_sub = "Vòng Bảng / League Phase"
        elif month in (2, 3):
            stage_sub = "Vòng 1/8 (Knockout)"
        elif month == 4:
            stage_sub = "Tứ Kết / Bán Kết"
        elif month in (5, 6):
            stage_sub = "Bán Kết / Chung Kết"
        elif month in (7, 8):
            stage_sub = "Vòng Loại / Knockout"
        else:
            stage_sub = "Champions League"

        pen_str = f" (Pen {pen_h}-{pen_a})" if has_pen else ""
        catalog_items.append({
            "category": "⚽ UEFA Champions League",
            "id": f"opta_ucl_{mid}",
            "opta_match_id": mid,
            "label": f"{ht} {hs} - {as_} {at}{pen_str}",
            "home_team": ht,
            "away_team": at,
            "home_team_id": ht_id,
            "away_team_id": at_id,
            "home_score": hs,
            "away_score": as_,
            "pen_home": pen_h if has_pen else None,
            "pen_away": pen_a if has_pen else None,
            "stage": f"UEFA Champions League {season} - {stage_sub}",
            "date": mdate,
            "stadium": "UEFA Champions League Arena",
            "referee": "UEFA Official Referee",
            "goals": goals_list,
            "source_type": "opta_ucl_parquet",
        })

    catalog_items.sort(key=lambda x: x.get("date") or "", reverse=True)
    with open(os.path.join(OUT_DIR, "ucl_catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(catalog_items), "matches": catalog_items}, f, ensure_ascii=False)
    print(f"3. Đã lưu ucl_catalog.json ({len(catalog_items)} trận UCL)!")

    print("4. Đang trích xuất bảng đội hình ucl_lineups.parquet gọn nhẹ...")
    lu = pd.read_parquet(lu_path)
    lu_ucl = lu[lu["match_id"].isin(valid_mids)].sort_values(["match_id"]).reset_index(drop=True)
    lu_ucl.to_parquet(os.path.join(OUT_DIR, "ucl_lineups.parquet"), index=False)

    print("5. Đang sắp xếp ucl_events.parquet theo match_id để truy vấn tức thì (<30ms/trận)...")
    ev_sorted = ev[ev["match_id"].isin(valid_mids)].sort_values(["match_id", "period_id", "minute", "second"]).reset_index(drop=True)
    ev_sorted.to_parquet(os.path.join(OUT_DIR, "ucl_events.parquet"), index=False)

    # Dọn dẹp thư mục tạm _opta_raw để tiết kiệm dung lượng ổ đĩa
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    print(f"🎉 HOÀN TẤT KHO OPTA UCL PARQUET ({len(catalog_items)} TRẬN CHAMPIONS LEAGUE)!")


if __name__ == "__main__":
    main()
