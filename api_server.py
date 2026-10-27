import json
import math
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "C:/Users/Admin/hdfs-practice"
if not os.path.exists(DATA_DIR):
    DATA_DIR = BASE_DIR
DB_PATH = os.path.join(BASE_DIR, "sofascore_platform.db")

_API_CACHE = {}


def _load_api_datasets():
    if _API_CACHE.get("loaded"):
        return _API_CACHE

    stats_path = os.path.join(BASE_DIR, "worldcup_match_stats.csv")
    shots_path = os.path.join(BASE_DIR, "worldcup_2018_shots.csv")
    events_path = os.path.join(DATA_DIR, "worldcup_0_Extract.csv")

    stats_df = pd.read_csv(stats_path, low_memory=False) if os.path.exists(stats_path) else pd.DataFrame()
    shots_df = pd.read_csv(shots_path, low_memory=False) if os.path.exists(shots_path) else pd.DataFrame()

    matches_meta = pd.DataFrame()
    if os.path.exists(events_path):
        try:
            use_cols = [
                "match_id", "match_date", "competition_stage_name",
                "home_team_home_team_name", "home_team_home_team_group",
                "away_team_away_team_name", "away_team_away_team_group",
                "home_score", "away_score", "stadium_name", "referee_name",
            ]
            matches_meta = (
                pd.read_csv(events_path, usecols=lambda c: c in use_cols, low_memory=False)
                .drop_duplicates(subset=["match_id"])
                .sort_values("match_date", ascending=False)
            )
        except Exception:
            matches_meta = pd.DataFrame()

    manifest_path = os.path.join(BASE_DIR, "sample_data", "statsbomb_catalog.json")
    sb_manifest_matches = []
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                sb_manifest_matches = json.load(mf).get("matches", [])
        except Exception:
            sb_manifest_matches = []

    ucl_manifest_path = os.path.join(BASE_DIR, "sample_data", "opta_ucl", "ucl_catalog.json")
    ucl_manifest_matches = []
    if os.path.isfile(ucl_manifest_path):
        try:
            with open(ucl_manifest_path, "r", encoding="utf-8") as uf:
                ucl_manifest_matches = json.load(uf).get("matches", [])
        except Exception:
            ucl_manifest_matches = []

    _API_CACHE["stats_df"] = stats_df
    _API_CACHE["shots_df"] = shots_df
    _API_CACHE["matches_meta"] = matches_meta
    _API_CACHE["sb_manifest_matches"] = sb_manifest_matches
    _API_CACHE["ucl_manifest_matches"] = ucl_manifest_matches
    _API_CACHE["loaded"] = True
    return _API_CACHE


def compute_wc2018_standings_data(matches_meta: pd.DataFrame):
    if matches_meta is None or matches_meta.empty:
        return {}
    grp_df = matches_meta[matches_meta["competition_stage_name"] == "Group Stage"].copy()
    groups = {}
    for _, row in grp_df.iterrows():
        g_name = str(row.get("home_team_home_team_group") or row.get("away_team_away_team_group") or "Group").strip()
        ht = str(row["home_team_home_team_name"])
        at = str(row["away_team_away_team_name"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        groups.setdefault(g_name, {})
        for t in (ht, at):
            groups[g_name].setdefault(t, {
                "team": t, "mp": 0, "w": 0, "d": 0, "l": 0,
                "gf": 0, "ga": 0, "gd": 0, "pts": 0, "form": []
            })
        h_rec, a_rec = groups[g_name][ht], groups[g_name][at]
        h_rec["mp"] += 1
        a_rec["mp"] += 1
        h_rec["gf"] += hs
        h_rec["ga"] += as_
        a_rec["gf"] += as_
        a_rec["ga"] += hs
        if hs > as_:
            h_rec["w"] += 1
            h_rec["pts"] += 3
            h_rec["form"].append("W")
            a_rec["l"] += 1
            a_rec["form"].append("L")
        elif hs < as_:
            a_rec["w"] += 1
            a_rec["pts"] += 3
            a_rec["form"].append("W")
            h_rec["l"] += 1
            h_rec["form"].append("L")
        else:
            h_rec["d"] += 1
            a_rec["d"] += 1
            h_rec["pts"] += 1
            a_rec["pts"] += 1
            h_rec["form"].append("D")
            a_rec["form"].append("D")

    result = {}
    for g_name in sorted(groups.keys()):
        rows = list(groups[g_name].values())
        for r in rows:
            r["gd"] = r["gf"] - r["ga"]
            r["form_str"] = "-".join(r["form"][-3:])
        rows.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
        for idx, r in enumerate(rows, 1):
            r["rank"] = idx
            r["qualified"] = idx <= 2
        result[g_name] = rows
    return result


def _poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match_ai_payload(home_team: str, away_team: str, stats_df: pd.DataFrame):
    h_xg_avg, h_xga_avg = 1.45, 1.15
    a_xg_avg, a_xga_avg = 1.35, 1.20
    if stats_df is not None and not stats_df.empty and "team_name" in stats_df.columns:
        for t_name, is_h in [(home_team, True), (away_team, False)]:
            t_rows = stats_df[stats_df["team_name"].astype(str).str.lower() == t_name.strip().lower()]
            if not t_rows.empty:
                xg_for = float(pd.to_numeric(t_rows["xg"], errors="coerce").mean())
                mids = t_rows["match_id"].tolist()
                opp_rows = stats_df[(stats_df["match_id"].isin(mids)) & (stats_df["team_name"] != t_rows.iloc[0]["team_name"])]
                xg_ag = float(pd.to_numeric(opp_rows["xg"], errors="coerce").mean()) if not opp_rows.empty else 1.15
                if is_h:
                    h_xg_avg, h_xga_avg = max(0.45, xg_for), max(0.45, xg_ag)
                else:
                    a_xg_avg, a_xga_avg = max(0.45, xg_for), max(0.45, xg_ag)

    lam_h = round(max(0.35, min(3.6, (h_xg_avg * a_xga_avg / 1.22) * 1.04)), 2)
    lam_a = round(max(0.35, min(3.6, (a_xg_avg * h_xga_avg / 1.22) * 0.97)), 2)

    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    score_probs = []
    for i in range(7):
        for j in range(7):
            p = _poisson_pmf(i, lam_h) * _poisson_pmf(j, lam_a)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i <= 5 and j <= 5:
                score_probs.append({"score": f"{i}-{j}", "home_goals": i, "away_goals": j, "prob_pct": round(p * 100, 2)})

    tot = p_home + p_draw + p_away or 1.0
    score_probs.sort(key=lambda x: x["prob_pct"], reverse=True)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "expected_goals_home": lam_h,
        "expected_goals_away": lam_a,
        "win_prob_home_pct": round(p_home / tot * 100, 1),
        "draw_prob_pct": round(p_draw / tot * 100, 1),
        "win_prob_away_pct": round(p_away / tot * 100, 1),
        "top_scorelines": score_probs[:6],
    }


class FootballAnalyticsAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Keep console clean

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)
        data = _load_api_datasets()

        if path in ("/", "/api", "/api/health"):
            db_users, db_favs = 0, 0
            if os.path.exists(DB_PATH):
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        db_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                        db_favs = conn.execute("SELECT COUNT(*) FROM favorites").fetchone()[0]
                except Exception:
                    pass
            sb_cnt = int(len(data.get("sb_manifest_matches", []))) + 5
            ucl_cnt = int(len(data.get("ucl_manifest_matches", [])))
            wc18_cnt = int(len(data["matches_meta"]))
            self._send_json({
                "status": "ok",
                "service": "SofaScore Big Data & Football Analytics REST API",
                "version": "2.5.0",
                "total_platform_matches": wc18_cnt + sb_cnt + ucl_cnt,
                "wc2018_matches": wc18_cnt,
                "statsbomb_json_matches": sb_cnt,
                "opta_ucl_parquet_matches": ucl_cnt,
                "wc2018_shots": int(len(data["shots_df"])),
                "sqlite_db_connected": os.path.exists(DB_PATH),
                "registered_users": db_users,
                "saved_favorites": db_favs,
                "endpoints": [
                    "/api/health",
                    "/api/matches?limit=40&competition=UCL",
                    "/api/standings",
                    "/api/players/top?limit=15",
                    "/api/predict?home=France&away=Croatia",
                ],
            })
            return

        if path == "/api/matches":
            limit = int((qs.get("limit") or ["40"])[0])
            comp_filter = str((qs.get("competition") or [""])[0]).strip().lower()
            q_filter = str((qs.get("q") or [""])[0]).strip().lower()
            records = []
            for item in data.get("ucl_manifest_matches", []):
                records.append({
                    "match_id": item.get("opta_match_id"),
                    "competition": item.get("category"),
                    "date": item.get("date"),
                    "stage": item.get("stage"),
                    "home_team": item.get("home_team"),
                    "away_team": item.get("away_team"),
                    "home_score": item.get("home_score"),
                    "away_score": item.get("away_score"),
                    "stadium": item.get("stadium"),
                })
            for item in data.get("sb_manifest_matches", []):
                records.append({
                    "match_id": item.get("statsbomb_match_id"),
                    "competition": item.get("category"),
                    "date": item.get("date"),
                    "stage": item.get("stage"),
                    "home_team": item.get("home_team"),
                    "away_team": item.get("away_team"),
                    "home_score": item.get("home_score"),
                    "away_score": item.get("away_score"),
                    "stadium": item.get("stadium"),
                })
            for _, r in data["matches_meta"].iterrows():
                records.append({
                    "match_id": int(r["match_id"]),
                    "competition": "🏆 FIFA World Cup",
                    "date": str(r.get("match_date", ""))[:10],
                    "stage": f"World Cup 2018 - {r.get('competition_stage_name', '')}",
                    "home_team": str(r["home_team_home_team_name"]),
                    "away_team": str(r["away_team_away_team_name"]),
                    "home_score": int(r["home_score"]),
                    "away_score": int(r["away_score"]),
                    "stadium": str(r.get("stadium_name", "")),
                })
            if comp_filter:
                alias_map = {
                    "ucl": "champions league",
                    "wc": "world cup",
                    "epl": "premier league",
                    "euro": "euro",
                }
                cf_norm = alias_map.get(comp_filter, comp_filter)
                records = [r for r in records if cf_norm in str(r.get("competition", "")).lower() or cf_norm in str(r.get("stage", "")).lower()]
            if q_filter:
                records = [r for r in records if q_filter in f"{r.get('home_team','')} {r.get('away_team','')} {r.get('stage','')}".lower()]
            records.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
            total_matched = len(records)
            records = records[:limit]
            self._send_json({"total_available": total_matched, "count": len(records), "matches": records})
            return

        if path == "/api/standings":
            standings = compute_wc2018_standings_data(data["matches_meta"])
            self._send_json({"competition": "FIFA World Cup 2018", "groups": standings})
            return

        if path == "/api/players/top":
            limit = int((qs.get("limit") or ["15"])[0])
            sh = data["shots_df"]
            if sh.empty:
                self._send_json({"players": []})
                return
            reg = sh[sh["period"].isin([1, 2, 3, 4])].copy()
            reg["is_goal"] = (reg["shot_outcome_name"] == "Goal").astype(int)
            reg["xg"] = pd.to_numeric(reg["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
            grp = (
                reg.groupby(["player_name", "team_name"], as_index=False)
                .agg(goals=("is_goal", "sum"), shots=("is_goal", "count"), xg=("xg", "sum"))
                .sort_values(["goals", "xg"], ascending=[False, False])
                .head(limit)
            )
            grp["xg"] = grp["xg"].round(2)
            self._send_json({"players": grp.to_dict(orient="records")})
            return

        if path == "/api/predict":
            home = (qs.get("home") or ["France"])[0]
            away = (qs.get("away") or ["Croatia"])[0]
            self._send_json(predict_match_ai_payload(home, away, data["stats_df"]))
            return

        self._send_json({"error": "Not Found", "path": path}, status=404)


_SERVER_THREAD = None


def ensure_api_server_running(port: int = 8502) -> bool:
    global _SERVER_THREAD
    if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
        return True
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), FootballAnalyticsAPIHandler)
        server.daemon_threads = True
        t = threading.Thread(target=server.serve_forever, name="FootballAnalyticsAPI", daemon=True)
        t.start()
        _SERVER_THREAD = t
        return True
    except OSError:
        # Port already bound by an active instance
        return True


if __name__ == "__main__":
    print("Starting Football Analytics REST API on http://localhost:8502 ...")
    srv = ThreadingHTTPServer(("0.0.0.0", 8502), FootballAnalyticsAPIHandler)
    srv.serve_forever()
