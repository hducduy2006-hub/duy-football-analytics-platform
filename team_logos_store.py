import json
import os
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CACHE_REL_PATH = os.path.join("sample_data", "pro_analytics", "team_logos_cache.json")
ESPN_CACHE_REL_PATH = os.path.join("sample_data", "pro_analytics", "espn_live_cache.json")
CACHE_TTL_SEC = 7 * 86400

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ESPN_TEAM_SLUGS = [
    "eng.1", "eng.2", "eng.3", "eng.4", "eng.fa", "eng.league_cup",
    "esp.1", "esp.2", "esp.copa_del_rey",
    "ita.1", "ita.2", "ita.coppa_italia",
    "ger.1", "ger.2", "ger.dfb_pokal",
    "fra.1", "fra.2", "fra.coupe_de_france",
    "uefa.champions", "uefa.europa", "uefa.europa.conf", "uefa.nations", "uefa.euro",
    "por.1", "ned.1", "sco.1", "sco.2", "tur.1", "tur.2",
    "bel.1", "aut.1", "sui.1", "gre.1", "nor.1", "den.1", "swe.1",
    "pol.1", "cze.1", "rou.1", "cro.1", "srb.1", "ukr.1", "hun.1",
    "bul.1", "isr.1", "cyp.1", "svk.1", "svn.1", "fin.1", "irl.1",
    "isl.1", "aze.1", "kaz.1",
    "fifa.world", "fifa.friendly", "fifa.worldq.uefa", "fifa.worldq.conmebol",
    "fifa.worldq.concacaf", "fifa.worldq.afc", "fifa.worldq.caf", "fifa.worldq.ofc",
    "conmebol.america", "concacaf.gold", "caf.nations", "afc.cup",
    "bra.1", "arg.1", "usa.1", "mex.1", "ksa.1", "jpn.1", "aus.1",
]

_CLUB_AFFIXES = {
    "fc", "cf", "sc", "ac", "fk", "sk", "sfk", "afc", "rc", "cd", "ud",
    "sd", "ca", "as", "ss", "us", "sv", "tsg", "vfl", "vfb", "rb", "bk",
    "if", "nk", "ks", "gnk", "pae", "ogc", "osc", "hsc", "aj", "estac",
    "rcd", "ca", "bv", "svw", "bsc", "tsv", "fsv", "spvgg", "1",
}

_AMBIGUOUS_SHORT_KEYS = {
    "sparta", "inter", "real", "athletic", "united", "city", "sporting",
    "dynamo", "dinamo", "racing", "olympique", "fc", "cf", "ac", "sc",
    "union", "victoria", "vitoria", "hapoel", "cska", "rapid", "lokomotiv",
    "red star", "atletico", "borussia", "eintracht", "fortuna", "stade",
}

_LOGO_ALIASES = {
    "az": "az alkmaar",
    "amed": "amed sfk",
    "amedspor": "amed sfk",
    "crvena zvezda": "red star belgrade",
    "fk crvena zvezda": "red star belgrade",
    "deportivo la coruna": "deportivo",
    "rc deportivo": "deportivo",
    "deportivo de la coruna": "deportivo",
    "egnatia rrogozhine": "egnatia",
    "erzurumspor": "erzurum bb",
    "erzurumspor fk": "erzurum bb",
    "gent": "kaa gent",
    "hapoel be er sheva": "hapoel be er",
    "hapoel beer sheva": "hapoel be er",
    "inter club d escaldes": "inter d escaldes",
    "kairat": "kairat almaty",
    "kups": "kups kuopio",
    "koln": "fc cologne",
    "1 fc koln": "fc cologne",
    "fc koln": "fc cologne",
    "lask": "lask linz",
    "ofi": "ofi crete",
    "olympiakos piraeus": "olympiacos",
    "olympiakos": "olympiacos",
    "rizespor": "caykur rizespor",
    "salzburg": "rb salzburg",
    "fc salzburg": "rb salzburg",
    "red bull salzburg": "rb salzburg",
    "sint truiden": "sint truidense",
    "slavia praha": "slavia prague",
    "sparta praha": "sparta prague",
    "union saint gilloise": "union st gilloise",
    "royale union saint gilloise": "union st gilloise",
    "universitatea craiova": "csu craiova",
    "cs universitatea craiova": "csu craiova",
    "vitoria guimaraes": "vitoria de guimaraes",
    "vitoria sc": "vitoria de guimaraes",
    "china pr": "china",
    "cote d ivoire": "ivory coast",
    "ir iran": "iran",
    "korea republic": "south korea",
    "turkey": "turkiye",
    "turkiye": "turkey",
    "czechia": "czech republic",
    "czech republic": "czechia",
    "usa": "united states",
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "fc internazionale milano": "inter milan",
    "ac milan": "ac milan",
    "milan": "ac milan",
    "atletico madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "athletic bilbao": "athletic club",
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "bayer leverkusen": "bayer leverkusen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "borussia monchengladbach": "borussia monchengladbach",
    "borussia m gladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "stade brestois": "brest",
    "cape verde islands": "cape verde",
    "stade rennais": "rennes",
    "hamburg sv": "hamburger sv",
    "hamburger sv": "hamburg sv",
    "mainz 05": "mainz",
    "1 fsv mainz 05": "mainz",
    "racing santander": "racing santander",
    "racing de santander": "racing santander",
    "istanbul basaksehir": "istanbul basaksehir",
    "corum": "corum fk",
    "bodo glimt": "bodo glimt",
    "fk bodo glimt": "bodo glimt",
    "nottingham": "nottingham forest",
    "nott m forest": "nottingham forest",
    "west brom": "west bromwich albion",
    "sheff utd": "sheffield united",
    "sheff wed": "sheffield wednesday",
    "brighton": "brighton hove albion",
    "newcastle": "newcastle united",
    "leicester": "leicester city",
    "leeds": "leeds united",
    "ipswich": "ipswich town",
    "norwich": "norwich city",
    "coventry": "coventry city",
    "luton": "luton town",
    "cardiff": "cardiff city",
    "swansea": "swansea city",
    "hull": "hull city",
    "stoke": "stoke city",
    "bristol city": "bristol city",
    "blackburn": "blackburn rovers",
    "preston": "preston north end",
    "qpr": "queens park rangers",
    "psv": "psv eindhoven",
    "feyenoord": "feyenoord",
    "ajax": "ajax",
    "sporting cp": "sporting cp",
    "sporting lisbon": "sporting cp",
    "fc porto": "fc porto",
    "porto": "fc porto",
    "benfica": "benfica",
    "sl benfica": "benfica",
    "braga": "sporting braga",
    "sc braga": "sporting braga",
    "alaves": "deportivo alaves",
    "celta vigo": "celta vigo",
    "real betis": "real betis",
    "betis": "real betis",
    "espanyol": "espanyol",
    "rcd espanyol": "espanyol",
    "mallorca": "mallorca",
    "rcd mallorca": "mallorca",
    "real sociedad": "real sociedad",
    "villarreal": "villarreal",
    "valencia": "valencia",
    "sevilla": "sevilla",
    "getafe": "getafe",
    "girona": "girona",
    "osasuna": "osasuna",
    "las palmas": "las palmas",
    "rayo vallecano": "rayo vallecano",
    "leganes": "leganes",
    "real valladolid": "real valladolid",
    "levante": "levante",
    "elche": "elche",
    "oviedo": "real oviedo",
    "mirandes": "mirandes",
    "almeria": "almeria",
    "granada": "granada",
    "cadiz": "cadiz",
    "eibar": "eibar",
    "malaga": "malaga",
    "marseille": "marseille",
    "olympique de marseille": "marseille",
    "olympique marseille": "marseille",
    "lyon": "lyon",
    "olympique lyonnais": "lyon",
    "monaco": "as monaco",
    "lille": "lille",
    "losc lille": "lille",
    "nice": "nice",
    "ogc nice": "nice",
    "lens": "rc lens",
    "strasbourg": "strasbourg",
    "reims": "stade de reims",
    "brest": "brest",
    "nantes": "nantes",
    "toulouse": "toulouse",
    "auxerre": "auxerre",
    "angers": "angers",
    "le havre": "le havre",
    "saint etienne": "saint etienne",
    "st etienne": "saint etienne",
    "montpellier": "montpellier",
    "le mans": "le mans",
    "roma": "as roma",
    "lazio": "lazio",
    "napoli": "napoli",
    "juventus": "juventus",
    "atalanta": "atalanta",
    "fiorentina": "fiorentina",
    "bologna": "bologna",
    "torino": "torino",
    "udinese": "udinese",
    "genoa": "genoa",
    "cagliari": "cagliari",
    "lecce": "lecce",
    "parma": "parma",
    "como": "como",
    "venezia": "venezia",
    "empoli": "empoli",
    "monza": "monza",
    "verona": "hellas verona",
    "frosinone": "frosinone",
    "sassuolo": "sassuolo",
    "pisa": "pisa",
    "spezia": "spezia",
    "cremonese": "cremonese",
    "palermo": "palermo",
    "sampdoria": "sampdoria",
    "galatasaray": "galatasaray",
    "fenerbahce": "fenerbahce",
    "besiktas": "besiktas",
    "trabzonspor": "trabzonspor",
    "celtic": "celtic",
    "rangers": "rangers",
    "aberdeen": "aberdeen",
    "hearts": "heart of midlothian",
    "hibernian": "hibernian",
    "stuttgart": "vfb stuttgart",
    "dortmund": "borussia dortmund",
    "leipzig": "rb leipzig",
    "frankfurt": "eintracht frankfurt",
    "freiburg": "sc freiburg",
    "hoffenheim": "tsg hoffenheim",
    "wolfsburg": "vfl wolfsburg",
    "werder bremen": "werder bremen",
    "augsburg": "fc augsburg",
    "union berlin": "1 fc union berlin",
    "st pauli": "fc st pauli",
    "heidenheim": "1 fc heidenheim",
    "bochum": "vfl bochum",
    "holstein kiel": "holstein kiel",
    "hertha berlin": "hertha bsc",
    "hertha bsc": "hertha berlin",
    "schalke 04": "schalke 04",
    "kaiserslautern": "1 fc kaiserslautern",
    "kiel": "holstein kiel",
}

_MEM_CACHE: dict | None = None


def normalize_logo_key(name: str) -> str:
    """Normalize a team name into a clean ASCII lowercase alphanumeric space-separated key."""
    s = str(name or "").strip()
    if not s:
        return ""
    s = (
        s.replace("ı", "i")
        .replace("İ", "I")
        .replace("ø", "o")
        .replace("Ø", "O")
        .replace("ł", "l")
        .replace("Ł", "L")
        .replace("đ", "d")
        .replace("Đ", "D")
        .replace("ß", "ss")
        .replace("æ", "ae")
        .replace("Æ", "AE")
        .replace("œ", "oe")
        .replace("Œ", "OE")
    )
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s


def _strip_club_affixes(norm_key: str) -> str:
    if not norm_key:
        return ""
    tokens = [tok for tok in norm_key.split() if tok not in _CLUB_AFFIXES]
    return " ".join(tokens).strip()


def _clean_espn_display_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+(?:FC|CF|AFC|SC|AC|EC)$", "", s, flags=re.IGNORECASE).strip()
    return s


def _fetch_espn_slug_teams(slug: str) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams?limit=200"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    out = []
    for sp in data.get("sports") or []:
        if not isinstance(sp, dict):
            continue
        for lg in sp.get("leagues") or []:
            if not isinstance(lg, dict):
                continue
            for item in lg.get("teams") or []:
                tm = item.get("team") if isinstance(item, dict) else None
                if not isinstance(tm, dict):
                    continue
                logos = tm.get("logos") or []
                logo = ""
                if logos and isinstance(logos[0], dict):
                    logo = str(logos[0].get("href") or "").strip()
                if not logo:
                    continue
                out.append({
                    "id": str(tm.get("id") or ""),
                    "displayName": str(tm.get("displayName") or ""),
                    "shortDisplayName": str(tm.get("shortDisplayName") or ""),
                    "name": str(tm.get("name") or ""),
                    "nickname": str(tm.get("nickname") or ""),
                    "abbreviation": str(tm.get("abbreviation") or ""),
                    "logo": logo,
                    "slug": slug,
                })
    return out


def _index_logo_candidate(logos_map: dict, candidate: str, logo_url: str, *, overwrite: bool = False, allow_short: bool = False):
    if not candidate or not logo_url:
        return
    nk = normalize_logo_key(candidate)
    if not nk:
        return
    stripped = _strip_club_affixes(nk)
    for k in (nk, stripped):
        if not k:
            continue
        if len(k) < 3 and not allow_short:
            continue
        if k in _AMBIGUOUS_SHORT_KEYS and k != nk:
            continue
        if k in _AMBIGUOUS_SHORT_KEYS and not overwrite and k in logos_map:
            continue
        if overwrite or k not in logos_map:
            logos_map[k] = logo_url


def _cache_abs_path() -> str:
    return os.path.join(_BASE_DIR, CACHE_REL_PATH)


def _espn_cache_abs_path() -> str:
    return os.path.join(_BASE_DIR, ESPN_CACHE_REL_PATH)


def build_team_logos_cache(force: bool = False) -> dict:
    """Fetch and persist ESPN club & national team HD logo mapping."""
    global _MEM_CACHE
    cache_path = _cache_abs_path()

    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("logos"), dict) and payload["logos"]:
                ts = float(payload.get("updated_ts") or 0.0)
                if (time.time() - ts) < CACHE_TTL_SEC:
                    _MEM_CACHE = payload
                    return payload
        except Exception:
            pass

    logos_map: dict[str, str] = {}
    fetched_teams: list[dict] = []

    with ThreadPoolExecutor(max_workers=16) as pool:
        for batch in pool.map(_fetch_espn_slug_teams, ESPN_TEAM_SLUGS):
            fetched_teams.extend(batch)

    # First pass: prioritize primary full displayName so exact clubs claim canonical names first
    for tm in fetched_teams:
        logo = tm["logo"]
        dn = tm["displayName"]
        cleaned = _clean_espn_display_name(dn)
        _index_logo_candidate(logos_map, dn, logo, overwrite=False, allow_short=True)
        if cleaned and cleaned != dn:
            _index_logo_candidate(logos_map, cleaned, logo, overwrite=False, allow_short=False)

    # Second pass: secondary names (shortDisplayName, name, nickname) without clobbering existing exact keys
    for tm in fetched_teams:
        logo = tm["logo"]
        for cand in (tm.get("shortDisplayName", ""), tm.get("name", ""), tm.get("nickname", "")):
            cand_s = str(cand or "").strip()
            if not cand_s:
                continue
            nk = normalize_logo_key(cand_s)
            if nk in _AMBIGUOUS_SHORT_KEYS or len(nk) < 3:
                continue
            _index_logo_candidate(logos_map, cand_s, logo, overwrite=False, allow_short=False)

    # Third pass: merge live scoreboard logos from espn_live_cache.json
    espn_live_path = _espn_cache_abs_path()
    if os.path.exists(espn_live_path):
        try:
            with open(espn_live_path, "r", encoding="utf-8") as fh:
                espn_data = json.load(fh)
            for fx in espn_data.get("live_fixtures") or []:
                if not isinstance(fx, dict):
                    continue
                for nm_k, lg_k in (("home_team", "home_logo"), ("away_team", "away_logo")):
                    nm = str(fx.get(nm_k) or "").strip()
                    lg = str(fx.get(lg_k) or "").strip()
                    if nm and lg:
                        _index_logo_candidate(logos_map, nm, lg, overwrite=False, allow_short=True)
        except Exception:
            pass

    # Fourth pass: propagate alias keys into cache for O(1) lookup
    for src_k, tgt_k in _LOGO_ALIASES.items():
        if src_k not in logos_map and tgt_k in logos_map:
            logos_map[src_k] = logos_map[tgt_k]
        elif tgt_k not in logos_map and src_k in logos_map:
            logos_map[tgt_k] = logos_map[src_k]

    payload = {
        "updated_ts": time.time(),
        "slugs_count": len(ESPN_TEAM_SLUGS),
        "teams_fetched": len(fetched_teams),
        "keys_count": len(logos_map),
        "logos": logos_map,
    }

    if logos_map:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except Exception:
            pass

    _MEM_CACHE = payload
    return payload


def _ensure_loaded_map() -> dict[str, str]:
    global _MEM_CACHE
    if isinstance(_MEM_CACHE, dict) and isinstance(_MEM_CACHE.get("logos"), dict):
        return _MEM_CACHE["logos"]

    cache_path = _cache_abs_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("logos"), dict) and payload["logos"]:
                _MEM_CACHE = payload
                return payload["logos"]
        except Exception:
            pass

    payload = build_team_logos_cache(force=False)
    return payload.get("logos", {}) if isinstance(payload, dict) else {}


def get_team_logo_url(team_name: str, fallback_url: str = "") -> str:
    """Resolve an HD PNG logo URL for any club or national team."""
    fb = str(fallback_url or "").strip()
    if fb.startswith(("http://", "https://")):
        return fb

    nm = str(team_name or "").strip()
    if not nm:
        return ""

    logos_map = _ensure_loaded_map()
    if not logos_map:
        return ""

    nk = normalize_logo_key(nm)
    candidates = []
    if nk:
        candidates.append(nk)
        if nk in _LOGO_ALIASES:
            candidates.append(_LOGO_ALIASES[nk])
        stripped = _strip_club_affixes(nk)
        if stripped and stripped != nk:
            candidates.append(stripped)
            if stripped in _LOGO_ALIASES:
                candidates.append(_LOGO_ALIASES[stripped])
        cleaned = normalize_logo_key(_clean_espn_display_name(nm))
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
            if cleaned in _LOGO_ALIASES:
                candidates.append(_LOGO_ALIASES[cleaned])

    for key in candidates:
        url = logos_map.get(key)
        if url:
            return url

    return ""


if __name__ == "__main__":
    data = build_team_logos_cache(force=True)
    print(f"Built team_logos_cache.json: fetched={data.get('teams_fetched')} keys={data.get('keys_count')}")

    up_path = os.path.join(_BASE_DIR, "sample_data", "pro_analytics", "upcoming_fixtures.parquet")
    if os.path.exists(up_path):
        import pandas as pd
        df = pd.read_parquet(up_path)
        up_teams = sorted(set(df["home_team"].dropna().astype(str)) | set(df["away_team"].dropna().astype(str)))
        missing = [t for t in up_teams if not get_team_logo_url(t)]
        matched = len(up_teams) - len(missing)
        print(f"Upcoming coverage: {matched}/{len(up_teams)} ({100.0 * matched / max(1, len(up_teams)):.1f}%)")
        if missing:
            print("Missing upcoming:", missing)
