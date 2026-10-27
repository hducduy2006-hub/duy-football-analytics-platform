"""
Lay anh cau thu qua Wikipedia REST API /page/summary (khong bi rate-limit neu delay 1s).
Khac voi truy cap direct image URL, REST API co gioi han 500 req/hour => du dung.
"""
import sys, json, time, urllib.request, urllib.parse, unicodedata, os
sys.stdout.reconfigure(encoding='utf-8')

try:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYERS = list(PLAYER_PHOTOS_GENERATED.keys())
except ImportError:
    print("Khong tim thay player_photos_generated.py"); exit(1)

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def wiki_page_title(name):
    """Tao cac bien the Wikipedia page title de thu."""
    parts = name.split()
    no_acc = remove_accents(name)
    variants = [name, no_acc]
    if len(parts) > 2:
        short = f"{parts[0]} {parts[-1]}"
        variants += [short, remove_accents(short)]
    if len(parts) >= 3:
        short2 = f"{parts[0]} {parts[-2]}"
        variants += [short2, remove_accents(short2)]
    # De-duplicate
    seen, result = set(), []
    for v in variants:
        if v not in seen: seen.add(v); result.append(v)
    return result

def get_wiki_photo(player_name):
    """Lay thumbnail tu Wikipedia REST API /page/summary/{title}"""
    titles = wiki_page_title(player_name)
    headers = {
        "User-Agent": "DUY-Football-Analytics/1.0 (https://localhost; duy@example.com)",
        "Accept": "application/json"
    }
    for title in titles:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title, safe='')}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    # Kiem tra co phai trang nguoi khong (co section "born", "footballer"...)
                    desc = (data.get("description") or "").lower()
                    extract = (data.get("extract") or "").lower()
                    if not any(w in desc + extract for w in ["footballer", "soccer", "football player", "midfielder", "defender", "forward", "goalkeeper", "striker", "winger"]):
                        time.sleep(0.3)
                        continue
                    thumb = data.get("thumbnail", {})
                    img_url = thumb.get("source", "")
                    if img_url:
                        # Lay anh lon hon (thay 320 -> 150)
                        img_url = img_url.replace("/320px-", "/200px-").replace("/330px-", "/200px-")
                        return img_url, title
        except Exception:
            pass
        time.sleep(0.3)
    return None, None

def save_results(d, path="player_photos_wiki_rest.py"):
    lines = ['"""Anh cau thu tu Wikipedia REST API - khong sua tay."""\n\n', "PLAYER_PHOTOS_WIKI_REST = {\n"]
    for k, v in d.items():
        lines.append(f'    "{k.replace(chr(34), chr(92)+chr(34))}": "{v}",\n')
    lines.append("}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

found = {}
not_found = []
total = len(PLAYERS)

print(f"Bat dau Wikipedia REST API cho {total} cau thu...")
print(f"Delay 0.8s/request -> ~{total*0.8/60:.0f} phut")
print("-" * 60)

for i, player in enumerate(PLAYERS):
    url, matched = get_wiki_photo(player)
    if url:
        found[player] = url
        if (i+1) % 15 == 0 or i < 5:
            print(f"  [{i+1}/{total}] OK ({matched}): {player}")
    else:
        not_found.append(player)

    time.sleep(0.8)  # 0.8s delay -> ~500 req/10 phut, khong bi rate limit

    if (i+1) % 50 == 0:
        save_results(found)
        print(f"  [SAVED {i+1}/{total}: {len(found)} found]")

save_results(found)
print(f"\n{'='*60}")
print(f"XONG: {len(found)}/{total} ({100*len(found)//total}%)")
print(f"Luu: player_photos_wiki_rest.py")
