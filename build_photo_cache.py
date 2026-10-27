"""
Build a local, verified photo cache for every player in the World Cup dataset.
Output: player_photos/<slug>.jpg (200x200) + player_photos/photo_index.json  {player_name: filename}
"""
import sys, os, re, io, json, time, unicodedata, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
PROJ = r"C:\Users\Admin\Downloads\spark-docker\spark-docker"
os.chdir(PROJ); sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "player_photos")
os.makedirs(OUT, exist_ok=True)

from player_photos_generated import PLAYER_PHOTOS_GENERATED as G
from player_photos_wiki_rest import PLAYER_PHOTOS_WIKI_REST as W
from player_photos_sportsdb import PLAYER_PHOTOS_SPORTSDB as S

UA = {"User-Agent": "DUYFootballAnalytics/1.0 (https://github.com/duy-analytics; hoangducduy.student@gmail.com) python-requests/2"}
sess = requests.Session(); sess.headers.update(UA)
lock = threading.Lock()

def strip(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()

def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", strip(name)).strip("_")

def fetch_img(url):
    try:
        for attempt in range(4):
            r = sess.get(url, timeout=25)
            if r.status_code in (403, 429, 503):
                time.sleep(2 * (attempt + 1)); continue
            break
        time.sleep(0.3)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = im.size
        if w < 60 or h < 60:
            return None
        side = min(w, h)
        left = (w - side) // 2
        top = 0 if h > w else (h - side) // 2       # portrait -> keep top (face)
        im = im.crop((left, top, left + side, top + side)).resize((200, 200), Image.LANCZOS)
        return im
    except Exception:
        return None

def std_thumb(url):
    # Wikimedia only serves standard thumbnail widths reliably -> force 330px
    return re.sub(r"/(\d+)px-", "/330px-", url) if url else url

def wiki_search(name, team):
    q = name.replace('"', "'")
    tokens = [t for t in re.split(r"[\s\-']+", strip(q)) if len(t) > 2]
    for query in (f"{q} footballer {team}", f"{q} footballer", " ".join(q.split()[:2]) + f" footballer {team}"):
        try:
            r = sess.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                "gsrlimit": 5, "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 330,
            }, timeout=20).json()
        except Exception:
            continue
        pages = sorted(r.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
        for p in pages:
            title = strip(p.get("title", ""))
            # title must share at least one meaningful name token with the player (avoid wrong person)
            if "thumbnail" in p and any(t in title for t in tokens):
                return p["thumbnail"]["source"]
    return None

players = pd.read_csv(r"C:\Users\Admin\hdfs-practice\worldcup_0_Extract.csv",
                      usecols=["player_name", "team_name"], low_memory=False).dropna().drop_duplicates("player_name")

index, stats = {}, {"ok": 0, "fail": []}

def work(row):
    name, team = row.player_name, row.team_name
    fname = slug(name) + ".jpg"
    fpath = os.path.join(OUT, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 2000:
        with lock: index[name] = fname; stats["ok"] += 1
        return
    candidates = [std_thumb(G.get(name)), std_thumb(W.get(name)), S.get(name)]
    im = None
    for u in filter(None, candidates):
        im = fetch_img(u)
        if im: break
    if im is None:
        u = wiki_search(name, team)
        if u: im = fetch_img(u)
    with lock:
        if im:
            im.save(fpath, "JPEG", quality=85); index[name] = fname; stats["ok"] += 1
        else:
            stats["fail"].append(name)

with ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(work, players.itertuples()))

with open(os.path.join(OUT, "photo_index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=1)
print(f"OK {stats['ok']}/{len(players)}  FAIL {len(stats['fail'])}")
print(stats["fail"])
