"""
Real Player Photographic Portrait Store & Prefetch Pipeline (v2)
Tier 1: FotMob 2026/27 Studio Cutouts (fresh 2026/27 full-face portraits, zero stale transfers)
Tier 2: EA Sports FC Official Men's Catalog (~16,228 studio PNG cutouts + name expansion)
Tier 3: Legacy local player_photos/ archive (799 World Cup / UCL legends)
Tier 4: Wikipedia REST/Action API (verified footballer portraits)
Tier 5: Broadcast Athlete Bust SVG fallback (bright studio backdrop + club kit + initials)
All resolved photos are rendered uncropped (full face, chin & collar) into 128x128 anti-aliased
circular RGBA WebP thumbnails on a clean bright pearl-silver studio backdrop.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import hashlib
import html
import io
import json
import os
import re
import threading
import time
import unicodedata
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "player_photos")
AVATARS_WEBP_DIR = os.path.join(PHOTO_DIR, "avatars_webp")
LEGACY_INDEX_PATH = os.path.join(PHOTO_DIR, "photo_index.json")
LIVE_CACHE_PATH = os.path.join(PHOTO_DIR, "live_avatar_cache.json")
EA_FC_CACHE_PATH = os.path.join(BASE_DIR, "sample_data", "pro_analytics", "ea_fc_portraits_cache.json")

os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(AVATARS_WEBP_DIR, exist_ok=True)

AVATAR_CACHE_VERSION = 2
EA_FC_API_URL = "https://drop-api.ea.com/rating/ea-sports-fc"
EA_FC_CATALOG_TTL = 14 * 86400
NEGATIVE_RETRY_TTL = 24 * 3600

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
_WIKI_HEADERS = {
    "User-Agent": "NCKH-TUD65-FootballAnalytics/1.0 (https://github.com/nckh-tud65; academic@tud65.edu)",
}

_CLUB_TEAM_ALIASES = {
    "ac milan": "milano fc",
    "milan": "milano fc",
    "inter milan": "lombardia fc",
    "inter": "lombardia fc",
    "internazionale": "lombardia fc",
    "lazio": "latium",
    "atalanta": "bergamo calcio",
    "bayern munich": "fc bayern munchen",
    "bayern": "fc bayern munchen",
    "psg": "paris sg",
    "paris saint germain": "paris sg",
    "manchester united": "manchester utd",
    "man utd": "manchester utd",
    "manchester city": "manchester city",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "atletico madrid": "atletico de madrid",
    "athletic bilbao": "athletic club",
    "barcelona": "fc barcelona",
    "villarreal": "villarreal cf",
    "levante": "levante ud",
    "elche": "elche cf",
    "frosinone": "frosinone",
    "getafe": "getafe cf",
    "osasuna": "ca osasuna",
    "malaga": "malaga cf",
    "monza": "ac monza",
    "fiorentina": "fiorentina",
    "marseille": "olympique marseille",
    "le mans": "le mans",
    "borussia dortmund": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "bayer leverkusen": "bayer 04 leverkusen",
    "leverkusen": "bayer 04 leverkusen",
    "wolverhampton wanderers": "wolves",
    "brighton hove albion": "brighton",
    "newcastle united": "newcastle utd",
    "nottingham forest": "nott m forest",
    "real betis": "real betis",
    "real sociedad": "real sociedad",
    "celta vigo": "rc celta",
    "istanbul basaksehir": "basaksehir",
    "bodo glimt": "bodo glimt",
}

_NAME_PARTICLES = {"de", "del", "da", "do", "dos", "das", "van", "von", "der", "den", "ten", "ter", "di", "la", "le", "el", "al", "bin", "ben"}

_LOCK = threading.RLock()
_EA_FC_INDEX: dict | None = None
_LEGACY_MAP: dict | None = None
_LIVE_CACHE: dict | None = None
_B64_MEM_CACHE: dict[str, str] = {}
_BG_TEMPLATE: Image.Image | None = None
_CIRCLE_MASK_128: Image.Image | None = None


def norm_name(s: str) -> str:
    txt = str(s or "").strip()
    if not txt:
        return ""
    txt = (
        txt.replace("ı", "i")
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
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[^a-z0-9]+", " ", txt.lower()).strip()
    return txt


def norm_compact_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm_name(s))


def _norm_team_aliases(team_name: str) -> set[str]:
    nk = norm_name(team_name)
    if not nk:
        return set()
    aliases = {nk}
    if nk in _CLUB_TEAM_ALIASES:
        aliases.add(_CLUB_TEAM_ALIASES[nk])
    for k, v in _CLUB_TEAM_ALIASES.items():
        if v == nk:
            aliases.add(k)
    # Strip generic club affixes for fuzzy matching
    toks = [t for t in nk.split() if t not in {"fc", "cf", "sc", "ac", "ud", "ca", "rc", "cd", "ssc", "as", "us", "rb", "vfl", "vfb", "tsg", "olympique", "sporting"}]
    if toks:
        aliases.add(" ".join(toks))
    return {a for a in aliases if len(a) >= 3}


def _team_match_score(player_team_norm: str, query_team_aliases: set[str]) -> int:
    if not player_team_norm or not query_team_aliases:
        return 0
    if player_team_norm in query_team_aliases:
        return 120
    p_toks = [t for t in player_team_norm.split() if t not in {"fc", "cf", "sc", "ac", "ud", "ca", "rc", "cd", "ssc", "as", "us", "de", "city", "united", "utd", "real"}]
    p_strip = " ".join(p_toks)
    for qa in query_team_aliases:
        if qa == p_strip or qa in player_team_norm or (len(player_team_norm) >= 4 and player_team_norm in qa):
            return 100
        if p_toks and any(pt in qa.split() and len(pt) >= 4 for pt in p_toks):
            return 60
    return 0


def _ensure_studio_templates() -> tuple[Image.Image, Image.Image]:
    global _BG_TEMPLATE, _CIRCLE_MASK_128
    if _BG_TEMPLATE is not None and _CIRCLE_MASK_128 is not None:
        return _BG_TEMPLATE, _CIRCLE_MASK_128

    size = 128
    bg = Image.new("RGBA", (size, size), (226, 232, 240, 255))
    px = bg.load()
    cx, cy = 64.0, 54.0
    for y in range(size):
        for x in range(size):
            d = min(1.0, (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) / 88.0)
            # Clean FotMob-style bright pearl-silver studio radial backdrop so every face & skin tone pops cleanly
            r = int(244 + (196 - 244) * (d ** 1.4))
            g = int(247 + (206 - 247) * (d ** 1.4))
            b = int(252 + (220 - 252) * (d ** 1.4))
            px[x, y] = (r, g, b, 255)

    # 4x supersampled anti-aliased circular mask
    mask_hi = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask_hi)
    draw.ellipse((4, 4, 507, 507), fill=255)
    mask_128 = mask_hi.resize((size, size), Image.Resampling.LANCZOS)

    _BG_TEMPLATE = bg
    _CIRCLE_MASK_128 = mask_128
    return _BG_TEMPLATE, _CIRCLE_MASK_128


def _process_image_bytes_to_circular_webp(raw_bytes: bytes, out_path: str) -> str | None:
    """Convert any headshot PNG/JPG/WebP into a crisp 128x128 uncropped full-face circular WebP thumbnail."""
    if not raw_bytes or len(raw_bytes) < 400:
        return None
    try:
        im = Image.open(io.BytesIO(raw_bytes))
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        w, h = im.size
        if w < 24 or h < 24:
            return None

        # Check corners to distinguish transparent cutouts (FotMob / EA FC) from opaque photos (Wiki / JPG)
        corners = [im.getpixel((0, 0))[3], im.getpixel((w - 1, 0))[3], im.getpixel((0, h - 1))[3], im.getpixel((w - 1, h - 1))[3]]
        is_cutout = (sum(1 for a in corners if a < 45) >= 3)

        if is_cutout:
            bbox = im.getbbox()
            if not bbox:
                return None
            trimmed = im.crop(bbox)
            bw, bh = trimmed.size
            if bw < 16 or bh < 16:
                return None
            # Preserve 100% of head, chin, neck, and upper collar inside the circle (zero chin clipping)
            side = max(int(round(max(bw, bh) * 1.12)), 48)
            canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            px_x = (side - bw) // 2
            px_y = max(int(round(side * 0.06)), side - bh)
            canvas.paste(trimmed, (px_x, px_y), trimmed)

            fg_128 = canvas.resize((128, 128), Image.Resampling.LANCZOS)
            bg_tmpl, mask_128 = _ensure_studio_templates()
            composed = bg_tmpl.copy()
            composed.alpha_composite(fg_128)
            composed.putalpha(mask_128)
        else:
            side = min(w, h)
            cx = w // 2
            left = max(0, min(w - side, cx - side // 2))
            top = 0 if h <= w else max(0, min(h - side, int((h - side) * 0.10)))
            sq = im.crop((left, top, left + side, top + side)).resize((128, 128), Image.Resampling.LANCZOS)
            _, mask_128 = _ensure_studio_templates()
            composed = sq
            composed.putalpha(mask_128)

        buf = io.BytesIO()
        composed.save(buf, format="WEBP", quality=86, method=4)
        webp_bytes = buf.getvalue()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(webp_bytes)
        return f"data:image/webp;base64,{base64.b64encode(webp_bytes).decode('ascii')}"
    except Exception:
        return None


def _file_to_data_uri(fpath: str) -> str | None:
    if not fpath or not os.path.exists(fpath):
        return None
    cached = _B64_MEM_CACHE.get(fpath)
    if cached:
        return cached
    try:
        with open(fpath, "rb") as fh:
            raw = fh.read()
        if not raw:
            return None
        uri = f"data:image/webp;base64,{base64.b64encode(raw).decode('ascii')}"
        _B64_MEM_CACHE[fpath] = uri
        return uri
    except Exception:
        return None


def _extract_ea_portrait_url(raw_url: str) -> str:
    s = str(raw_url or "").strip()
    if not s:
        return ""
    if s.startswith("http"):
        return s.split("?")[0]
    # Compact numeric/prefixed portrait ID -> full CDN URL
    return f"https://ratings-images-prod.pulse.ea.com/FC25/full/player-portraits/{s}.png"


def _compact_ea_portrait_id(raw_url: str) -> str:
    s = str(raw_url or "").strip().split("?")[0]
    prefix = "https://ratings-images-prod.pulse.ea.com/FC25/full/player-portraits/"
    if s.startswith(prefix) and s.endswith(".png"):
        return s[len(prefix):-4]
    return s


def _fetch_ea_fc_page(offset: int) -> tuple[int, list[list]]:
    url = f"{EA_FC_API_URL}?locale=en&limit=100&offset={offset}&gender=0"
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            total = int(data.get("totalItems") or 0)
            rows = []
            for it in data.get("items") or []:
                av = _compact_ea_portrait_id(it.get("avatarUrl") or "")
                if not av:
                    continue
                rows.append([
                    av,
                    str(it.get("firstName") or "").strip(),
                    str(it.get("lastName") or "").strip(),
                    str(it.get("commonName") or "").strip(),
                    str(((it.get("team") or {}).get("label")) or "").strip(),
                    int(it.get("overallRating") or 0),
                ])
            return total, rows
        except Exception:
            if attempt == 0:
                time.sleep(0.25)
    return 0, []


def build_ea_fc_catalog(force: bool = False) -> dict:
    """Download and persist all men's player studio portrait pointers from EA Sports FC API."""
    if not force and os.path.exists(EA_FC_CACHE_PATH):
        try:
            with open(EA_FC_CACHE_PATH, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("players") and (time.time() - float(d.get("updated_ts") or 0.0)) < EA_FC_CATALOG_TTL:
                return d
        except Exception:
            pass

    total, page0 = _fetch_ea_fc_page(0)
    all_rows: list[list] = list(page0)
    if total > 100:
        offsets = list(range(100, total, 100))
        with ThreadPoolExecutor(max_workers=16) as pool:
            for _t, batch in pool.map(_fetch_ea_fc_page, offsets):
                if batch:
                    all_rows.extend(batch)

    payload = {
        "updated_ts": time.time(),
        "total_reported": total,
        "count": len(all_rows),
        "players": all_rows,
    }
    if all_rows:
        os.makedirs(os.path.dirname(EA_FC_CACHE_PATH), exist_ok=True)
        with open(EA_FC_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    return payload


def _index_ea_fc_record(exact_map: dict, init_map: dict, surname_map: dict, idx_rec: int, rec: list, team_norm: str):
    _av, fn, ln, cn, _tm, ovr = rec
    item = (idx_rec, team_norm, int(ovr or 0))
    n_fn = norm_name(fn)
    n_ln = norm_name(ln)
    n_cn = norm_name(cn)
    full = f"{n_fn} {n_ln}".strip()

    if full:
        exact_map.setdefault(full, []).append((item, 45))
    if n_cn:
        exact_map.setdefault(n_cn, []).append((item, 44))
        cn_toks = n_cn.split()
        if len(cn_toks) >= 2:
            init_map.setdefault(f"{cn_toks[0][0]} {' '.join(cn_toks[1:])}", []).append((item, 24))
            init_map.setdefault(f"{cn_toks[0][0]} {cn_toks[-1]}", []).append((item, 20))
        else:
            surname_map.setdefault(n_cn, []).append((item, 15))

    f_toks = n_fn.split()
    l_toks = n_ln.split()
    if f_toks and l_toks:
        f0 = f_toks[0]
        # Full first + first surname token (e.g. Fermín López Marín -> fermin lopez)
        if len(l_toks) >= 2 and l_toks[0] not in _NAME_PARTICLES:
            exact_map.setdefault(f"{f0} {l_toks[0]}", []).append((item, 38))
        # Full first + last surname token (e.g. Diego Varela Pampín -> diego pampin)
        if len(l_toks) >= 2:
            exact_map.setdefault(f"{f0} {l_toks[-1]}", []).append((item, 36))
        # Initial + full surname (e.g. F. de Jong, A. Mac Allister)
        init_map.setdefault(f"{f0[0]} {n_ln}", []).append((item, 28))
        if len(l_toks) >= 2:
            if l_toks[0] not in _NAME_PARTICLES:
                init_map.setdefault(f"{f0[0]} {l_toks[0]}", []).append((item, 24))
            init_map.setdefault(f"{f0[0]} {l_toks[-1]}", []).append((item, 22))
    elif n_fn and not n_ln:
        surname_map.setdefault(n_fn, []).append((item, 14))

    if l_toks:
        if l_toks[0] not in _NAME_PARTICLES:
            surname_map.setdefault(l_toks[0], []).append((item, 10))
        surname_map.setdefault(l_toks[-1], []).append((item, 12))
        if len(l_toks) >= 2:
            surname_map.setdefault(n_ln, []).append((item, 16))


def _get_ea_fc_index() -> dict:
    global _EA_FC_INDEX
    with _LOCK:
        if _EA_FC_INDEX is not None:
            return _EA_FC_INDEX

        if os.path.exists(EA_FC_CACHE_PATH):
            try:
                with open(EA_FC_CACHE_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
        else:
            data = build_ea_fc_catalog(force=False)

        players = data.get("players") or []
        exact_map: dict[str, list] = {}
        init_map: dict[str, list] = {}
        surname_map: dict[str, list] = {}
        for i, rec in enumerate(players):
            t_norm = norm_name(rec[4]) if len(rec) > 4 else ""
            _index_ea_fc_record(exact_map, init_map, surname_map, i, rec, t_norm)

        _EA_FC_INDEX = {
            "players": players,
            "exact": exact_map,
            "init": init_map,
            "surname": surname_map,
        }
        return _EA_FC_INDEX


def _lookup_ea_fc_portrait(player_name: str, team_name: str = "") -> tuple[str, str, float]:
    """Return (portrait_url, resolved_full_name, confidence_score) from the 16k+ EA Sports FC catalog."""
    idx = _get_ea_fc_index()
    players = idx["players"]
    if not players:
        return "", "", 0.0

    np = norm_name(player_name)
    if not np:
        return "", "", 0.0
    toks = np.split()
    team_aliases = _norm_team_aliases(team_name)

    cand_scores: dict[int, float] = {}

    def _collect(bucket_map: dict, key: str, bonus: float = 0.0):
        for (rec_i, tm_n, ovr), base_sc in bucket_map.get(key, ()):
            t_sc = _team_match_score(tm_n, team_aliases)
            sc = base_sc + bonus + t_sc + (ovr / 10.0)
            if sc > cand_scores.get(rec_i, -1.0):
                cand_scores[rec_i] = sc

    # 1. Exact full / common name lookup
    _collect(idx["exact"], np, bonus=15.0)
    if len(toks) >= 3:
        _collect(idx["exact"], f"{toks[0]} {toks[-1]}", bonus=8.0)
        _collect(idx["exact"], f"{toks[0]} {toks[1]}", bonus=6.0)
        for j in range(1, len(toks)):
            _collect(idx["exact"], f"{toks[0]} {toks[j]}", bonus=4.0)

    # 2. Abbreviated Opta first-initial + surname (e.g. "F. de Jong", "L. Yamal", "A. Mac Allister")
    if len(toks) >= 2 and len(toks[0]) == 1:
        _collect(idx["init"], np, bonus=12.0)
        _collect(idx["init"], f"{toks[0]} {toks[-1]}", bonus=6.0)
    elif len(toks) >= 2:
        _collect(idx["init"], f"{toks[0][0]} {' '.join(toks[1:])}", bonus=2.0)
        _collect(idx["init"], f"{toks[0][0]} {toks[-1]}", bonus=0.0)
    elif len(toks) == 1:
        _collect(idx["surname"], toks[0], bonus=5.0)

    if not cand_scores:
        return "", "", 0.0

    best_i, best_sc = max(cand_scores.items(), key=lambda kv: kv[1])
    if best_sc < 18.0:
        return "", "", 0.0

    rec = players[best_i]
    url = _extract_ea_portrait_url(rec[0])
    full_disp = str(rec[3] or f"{rec[1]} {rec[2]}").strip()
    return url, full_disp, float(best_sc)


def _load_legacy_index() -> dict:
    global _LEGACY_MAP
    with _LOCK:
        if _LEGACY_MAP is not None:
            return _LEGACY_MAP

        by_key: dict[str, str] = {}
        by_init: dict[str, str] = {}
        if os.path.exists(LEGACY_INDEX_PATH):
            try:
                with open(LEGACY_INDEX_PATH, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except Exception:
                raw = {}
            for k, fn in raw.items():
                fp = fn if os.path.isabs(fn) else os.path.join(PHOTO_DIR, os.path.basename(fn))
                if not os.path.exists(fp):
                    continue
                np = norm_name(k)
                if np:
                    by_key.setdefault(np, fp)
                    by_key.setdefault(norm_compact_key(np), fp)
                    toks = np.split()
                    if len(toks) >= 2:
                        by_init.setdefault(f"{toks[0][0]} {toks[-1]}", fp)
                        by_init.setdefault(f"{toks[0]} {toks[-1]}", fp)
                        if len(toks) >= 3 and toks[1] not in _NAME_PARTICLES:
                            by_init.setdefault(f"{toks[0][0]} {toks[1]}", fp)

        if os.path.isdir(PHOTO_DIR):
            for fn in os.listdir(PHOTO_DIR):
                if os.path.splitext(fn)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    fp = os.path.join(PHOTO_DIR, fn)
                    stem = norm_name(os.path.splitext(fn)[0])
                    if stem:
                        by_key.setdefault(stem, fp)
                        by_key.setdefault(norm_compact_key(stem), fp)

        _LEGACY_MAP = {"by_key": by_key, "by_init": by_init}
        return _LEGACY_MAP


def _lookup_legacy_photo_file(player_name: str) -> str:
    idx = _load_legacy_index()
    np = norm_name(player_name)
    if not np:
        return ""
    for key in (np, norm_compact_key(np)):
        fp = idx["by_key"].get(key)
        if fp and os.path.exists(fp):
            return fp
    toks = np.split()
    if len(toks) >= 2:
        for key in (f"{toks[0]} {toks[-1]}", f"{toks[0][0]} {' '.join(toks[1:])}", f"{toks[0][0]} {toks[-1]}"):
            fp = idx["by_init"].get(key)
            if fp and os.path.exists(fp):
                return fp
    return ""


def _lookup_fotmob_portrait(player_name: str, team_name: str = "", surname_only: bool = False) -> tuple[str, str]:
    """Query FotMob public suggest endpoint for 2026/27 rookies/transfers not in EA FC."""
    np = norm_name(player_name)
    toks = np.split()
    if not toks:
        return "", ""

    team_aliases = _norm_team_aliases(team_name)
    if surname_only:
        if not team_aliases or len(toks[-1]) < 4:
            return "", ""
        query = toks[-1]
    else:
        query = player_name if len(toks[0]) > 1 else toks[-1]

    url = f"https://www.fotmob.com/api/data/search/suggest?term={urllib.parse.quote(query)}&lang=en"
    try:
        data = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=_HTTP_HEADERS), timeout=5).read().decode("utf-8"))
    except Exception:
        return "", ""

    best_url, best_name, best_sc = "", "", -1.0
    for grp in data or []:
        if not isinstance(grp, dict):
            continue
        for s in grp.get("suggestions") or []:
            if not isinstance(s, dict) or s.get("type") != "player" or not s.get("id"):
                continue
            cand_nm = str(s.get("name") or "").strip()
            cand_norm = norm_name(cand_nm)
            cand_toks = cand_norm.split()
            if not cand_toks:
                continue

            t_sc = _team_match_score(norm_name(s.get("teamName") or ""), team_aliases)
            if surname_only and t_sc < 60:
                continue

            sc = 0.0
            if cand_norm == np:
                sc += 60.0
            elif toks[-1] in cand_toks:
                sc += 35.0
            else:
                continue

            if len(toks) >= 2 and toks[0][0] == cand_toks[0][0]:
                sc += 20.0
                if len(toks[0]) > 1 and toks[0] == cand_toks[0]:
                    sc += 15.0

            sc += t_sc
            if sc > best_sc:
                best_sc = sc
                best_url = f"https://images.fotmob.com/image_resources/playerimages/{s['id']}.png"
                best_name = cand_nm

    if best_url and best_sc >= 35.0:
        return best_url, best_name
    return "", ""


def _lookup_wikipedia_portrait(player_name: str) -> str:
    """Query Wikipedia Action API (requiring football description) as last remote image fallback."""
    np = norm_name(player_name)
    toks = np.split()
    if not toks or len(toks[0]) <= 1:
        return ""
    try:
        params = urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{player_name} footballer",
            "gsrlimit": "3",
            "prop": "pageimages|description",
            "pithumbsize": "240",
            "pilicense": "any",
            "format": "json",
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        data = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=_WIKI_HEADERS), timeout=5).read().decode("utf-8"))
        for p in ((data.get("query") or {}).get("pages") or {}).values():
            thumb = (p.get("thumbnail") or {}).get("source", "")
            desc = str(p.get("description") or "").lower()
            title_n = norm_name(p.get("title") or "")
            if thumb and toks[-1] in title_n and any(k in desc for k in ("football", "soccer", "goalkeeper", "midfielder", "defender", "forward", "winger", "striker")):
                return thumb
    except Exception:
        pass
    return ""


def _wipe_stale_v1_avatars():
    _B64_MEM_CACHE.clear()
    if os.path.isdir(AVATARS_WEBP_DIR):
        for fn in os.listdir(AVATARS_WEBP_DIR):
            if fn.endswith(".webp"):
                try:
                    os.remove(os.path.join(AVATARS_WEBP_DIR, fn))
                except Exception:
                    pass


def _load_live_cache() -> dict:
    global _LIVE_CACHE
    with _LOCK:
        if _LIVE_CACHE is not None:
            return _LIVE_CACHE
        if os.path.exists(LIVE_CACHE_PATH):
            try:
                with open(LIVE_CACHE_PATH, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict) and d.get("__version__") == AVATAR_CACHE_VERSION:
                    _LIVE_CACHE = d
                    return d
            except Exception:
                pass
        # v1 or corrupt cache -> wipe old chin-cropped dark avatars once and initialize clean v2 map
        _wipe_stale_v1_avatars()
        _LIVE_CACHE = {"__version__": AVATAR_CACHE_VERSION}
        return _LIVE_CACHE


def _save_live_cache():
    with _LOCK:
        if _LIVE_CACHE is None:
            return
        try:
            os.makedirs(os.path.dirname(LIVE_CACHE_PATH), exist_ok=True)
            with open(LIVE_CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(_LIVE_CACHE, fh, ensure_ascii=False)
        except Exception:
            pass


def _avatar_cache_key(player_name: str, team_name: str = "") -> str:
    return f"{norm_name(player_name)}|{norm_name(team_name)}"


def _download_bytes(url: str, timeout: float = 7.0) -> bytes:
    if not url:
        return b""
    hdrs = _WIKI_HEADERS if "wikipedia" in url or "wikimedia" in url else _HTTP_HEADERS
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _resolve_and_store_single(player_name: str, team_name: str = "", allow_network: bool = True) -> tuple[str, bool]:
    """Resolve one player across all 4 photo tiers, save circular WebP, and return (data_uri, cache_dirty)."""
    if not player_name or str(player_name).strip().lower() in ("unknown", "none", "nan"):
        return "", False

    ck = _avatar_cache_key(player_name, team_name)
    live_cache = _load_live_cache()
    with _LOCK:
        rec = live_cache.get(ck)

    if isinstance(rec, dict):
        fn = rec.get("file")
        if fn:
            fpath = os.path.join(AVATARS_WEBP_DIR, os.path.basename(fn))
            uri = _file_to_data_uri(fpath)
            if uri:
                return uri, False
        elif rec.get("miss_ts") and (time.time() - float(rec["miss_ts"])) < NEGATIVE_RETRY_TTL:
            return "", False

    # Slug filename unique per player + team so different clubs/same-surname teammates stay separate
    slug_part = re.sub(r"[^a-z0-9_]", "", norm_name(player_name).replace(" ", "_"))[:32] or "player"
    h6 = hashlib.md5(ck.encode("utf-8")).hexdigest()[:8]
    out_fn = f"{slug_part}_{h6}.webp"
    out_path = os.path.join(AVATARS_WEBP_DIR, out_fn)

    # Check if exact WebP already on disk even if cache JSON lacked pointer
    existing_uri = _file_to_data_uri(out_path)
    if existing_uri:
        with _LOCK:
            live_cache[ck] = {"file": out_fn, "source": "disk"}
        return existing_uri, True

    uri = None
    source_tag = ""

    # Local index lookup (~0ms) gives full name expansion ("J. Alvarez" -> "Julian Alvarez") + EA FC cutout fallback
    ea_url, ea_disp, ea_sc = _lookup_ea_fc_portrait(player_name, team_name=team_name)
    is_initial_only = bool(norm_name(player_name).split() and len(norm_name(player_name).split()[0]) == 1)

    # Tier 1: FotMob 2026/27 fresh full-face cutouts
    if allow_network:
        fm_queries = []
        if is_initial_only and ea_disp and ea_sc >= 55.0:
            fm_queries.append((ea_disp, False))
        fm_queries.append((player_name, False))
        if team_name and len(norm_name(player_name).split()) >= 2:
            fm_queries.append((player_name, True))
        for fq, s_only in fm_queries:
            fm_url, _ = _lookup_fotmob_portrait(fq, team_name=team_name, surname_only=s_only)
            if not fm_url:
                continue
            try:
                raw = _download_bytes(fm_url, timeout=6.0)
                uri = _process_image_bytes_to_circular_webp(raw, out_path)
                if uri:
                    source_tag = "fotmob"
                    break
            except Exception:
                uri = None

    # Tier 2: EA Sports FC 16,228 studio cutouts (fallback when FotMob unavailable or 403/404)
    if not uri and ea_url and allow_network:
        try:
            raw = _download_bytes(ea_url, timeout=7.0)
            uri = _process_image_bytes_to_circular_webp(raw, out_path)
            if uri:
                source_tag = "eafc"
        except Exception:
            uri = None

    # Tier 3: Legacy local player_photos/ archive (offline World Cup / UCL legends)
    if not uri:
        legacy_fp = _lookup_legacy_photo_file(player_name)
        if legacy_fp:
            try:
                with open(legacy_fp, "rb") as fh:
                    raw = fh.read()
                uri = _process_image_bytes_to_circular_webp(raw, out_path)
                if uri:
                    source_tag = "legacy"
            except Exception:
                uri = None

    # Tier 4: Wikipedia footballer thumbnail fallback
    if not uri and allow_network:
        wiki_url = _lookup_wikipedia_portrait(player_name)
        if wiki_url:
            try:
                raw = _download_bytes(wiki_url, timeout=6.0)
                uri = _process_image_bytes_to_circular_webp(raw, out_path)
                if uri:
                    source_tag = "wikipedia"
            except Exception:
                uri = None

    with _LOCK:
        if uri:
            _B64_MEM_CACHE[out_path] = uri
            live_cache[ck] = {"file": out_fn, "source": source_tag}
            return uri, True
        elif allow_network:
            live_cache[ck] = {"file": "", "miss_ts": time.time()}
            return "", True

    return "", False


def prefetch_match_avatars(players_with_teams: list[tuple[str, str]], max_workers: int = 12) -> int:
    """Resolve and cache all missing player portraits for a match in parallel."""
    if not players_with_teams:
        return 0

    live_cache = _load_live_cache()
    _get_ea_fc_index()
    _load_legacy_index()

    to_fetch: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    now_t = time.time()

    for p_name, t_name in players_with_teams:
        pn = str(p_name or "").strip()
        tn = str(t_name or "").strip()
        if not pn:
            continue
        ck = _avatar_cache_key(pn, tn)
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        rec = live_cache.get(ck)
        if isinstance(rec, dict):
            fn = rec.get("file")
            if fn and os.path.exists(os.path.join(AVATARS_WEBP_DIR, os.path.basename(fn))):
                continue
            if rec.get("miss_ts") and (now_t - float(rec["miss_ts"])) < NEGATIVE_RETRY_TTL:
                continue
        to_fetch.append((pn, tn))

    if not to_fetch:
        return 0

    any_dirty = False
    resolved_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_resolve_and_store_single, pn, tn, True) for pn, tn in to_fetch]
        for fut in as_completed(futs):
            try:
                uri, dirty = fut.result()
                if uri:
                    resolved_count += 1
                if dirty:
                    any_dirty = True
            except Exception:
                continue

    if any_dirty:
        _save_live_cache()
    return resolved_count


def get_player_avatar_b64(player_name: str, team_name: str = "", allow_network: bool = True) -> str:
    """Return data:image/webp;base64,... circular studio portrait or empty string."""
    uri, dirty = _resolve_and_store_single(player_name, team_name=team_name, allow_network=allow_network)
    if dirty:
        _save_live_cache()
    return uri or ""


def make_broadcast_bust_svg(initials: str, primary_color: str, secondary_color: str, size: int = 80) -> str:
    """Broadcast-styled footballer silhouette avatar fallback matching bright pearl-silver studio circles."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="{size}" height="{size}">'
        f'<defs>'
        f'<radialGradient id="bbg" cx="50%" cy="42%" r="65%">'
        f'<stop offset="0%" stop-color="#F4F7FC"/><stop offset="100%" stop-color="#C4CEDC"/>'
        f'</radialGradient>'
        f'<linearGradient id="kit" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{primary_color}"/><stop offset="100%" stop-color="{primary_color}" stop-opacity="0.88"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<circle cx="40" cy="40" r="39" fill="url(#bbg)"/>'
        f'<circle cx="40" cy="29" r="12.5" fill="#475569"/>'
        f'<path d="M16 74 C18 54 27 46 40 46 C53 46 62 54 64 74 Z" fill="url(#kit)" stroke="{secondary_color}" stroke-width="2"/>'
        f'<text x="40" y="66" font-size="13.5" font-weight="900" text-anchor="middle" '
        f'fill="{secondary_color}" font-family="Inter,Arial,sans-serif">{html.escape(initials)}</text>'
        f'</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _prewarm_featured_matches():
    from prematch_research import load_espn_live_cache
    import daily_realtime_hub as drh
    import daily_match_center_bridge as dmb

    sample_dir = os.path.join(BASE_DIR, "sample_data")
    live_data = load_espn_live_cache()
    fixtures = live_data.get("live_fixtures") or []
    print(f"Pre-warming player portraits for {min(len(fixtures), 16)} live/recent ESPN fixtures...")

    pairs: list[tuple[str, str]] = []
    # 1. Explicitly check featured teams & recent Opta squads
    xi_index = dmb._load_latest_club_xi_index(sample_dir)
    target_teams = [
        "Barcelona", "Villarreal", "AC Milan", "Frosinone", "Levante", "Elche",
        "Getafe", "Atlético Madrid", "Atalanta", "Inter Milan", "Marseille", "Le Mans",
        "Osasuna", "Málaga", "Liverpool", "Manchester United", "Werder Bremen",
        "Hamburg SV", "Real Madrid", "Spain", "Argentina",
    ]
    for tn in target_teams:
        rec = xi_index.get(drh._team_alias_key(tn))
        if rec:
            for p_item in (rec.get("starters") or []) + (rec.get("bench") or []) + (rec.get("subs") or []):
                nm = p_item.get("name") if isinstance(p_item, dict) else (p_item[0] if isinstance(p_item, (list, tuple)) and p_item else str(p_item))
                if nm and "thủ môn" not in str(nm).lower() and "dự bị" not in str(nm).lower():
                    pairs.append((nm, tn))

    # 2. Live/post fixtures with ESPN rosters
    for fx in fixtures[:14]:
        eid = str(fx.get("espn_event_id") or "")
        slug = str(fx.get("espn_slug") or "")
        st_f = str(fx.get("state") or "")
        if eid and slug and st_f in ("in", "post"):
            try:
                summ = drh.get_espn_match_summary(slug, eid, force_refresh=False, fallback_match=fx)
                for ros in summ.get("rosters") or []:
                    t_nm = ros.get("team_name") or ""
                    for p in (ros.get("starters") or []) + (ros.get("bench") or []):
                        if p.get("name"):
                            pairs.append((p["name"], t_nm))
            except Exception:
                continue

    n_new = prefetch_match_avatars(pairs, max_workers=12)
    print(f"Pre-warmed {len(pairs)} roster slots ({n_new} newly downloaded circular WebP files)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build EA FC catalog and pre-warm featured matches")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cat = build_ea_fc_catalog(force=False)
    print(f"EA FC catalog ready: {cat.get('count')} men's players in {time.perf_counter()-t0:.1f}s")
    if args.build:
        _prewarm_featured_matches()
