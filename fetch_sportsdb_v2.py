"""
Version 2: Fetch player photos from TheSportsDB with smart name shortening.
Strips accents, tries multiple name variants to maximize hit rate.
"""
import sys, json, time, urllib.request, urllib.parse, unicodedata
sys.stdout.reconfigure(encoding='utf-8')

try:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYERS = list(PLAYER_PHOTOS_GENERATED.keys())
except ImportError:
    print("Khong tim thay player_photos_generated.py")
    exit(1)

def remove_accents(text):
    """Bo dau tieng nuoc ngoai: Müller -> Muller, Mbappé -> Mbappe"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def get_name_variants(full_name):
    """Tao nhieu bien the ten de tang kha nang tim thay."""
    parts = full_name.split()
    variants = []
    
    if len(parts) == 1:
        variants = [full_name, remove_accents(full_name)]
    elif len(parts) == 2:
        # "Harry Kane" -> thu chinh xac
        variants = [full_name, remove_accents(full_name)]
    elif len(parts) == 3:
        # "Kylian Mbappe Lottin" -> "Kylian Mbappe", "Mbappe Lottin"
        variants = [
            f"{parts[0]} {parts[1]}",          # First + Middle
            f"{parts[0]} {parts[-1]}",          # First + Last
            f"{parts[-2]} {parts[-1]}",         # Middle + Last (ex: "Mbappe Lottin")
            parts[-1],                           # Just last name
        ]
    else:
        # "Lionel Andres Messi Cuccittini" -> nhieu bien the
        variants = [
            f"{parts[0]} {parts[-1]}",          # First + Last: "Lionel Cuccittini" (sai)
            f"{parts[0]} {parts[-2]}",          # First + 2nd last: "Lionel Messi" (dung!)
            f"{parts[0]} {parts[1]}",           # First + 2nd: "Lionel Andres"
            f"{parts[-2]} {parts[-1]}",         # 2nd last + Last
            parts[-1],                           # Chi last
            parts[-2],                           # Chi 2nd last (thong thuong la ho)
            full_name,                           # Ten day du
        ]
    
    # Them bien the bo dau cho tung bien the
    expanded = []
    for v in variants:
        expanded.append(v)
        no_accent = remove_accents(v)
        if no_accent != v:
            expanded.append(no_accent)
    
    # Loai trung lap, giu thu tu
    seen = set()
    result = []
    for v in expanded:
        if v and v not in seen and len(v) > 2:
            seen.add(v)
            result.append(v)
    return result

def _save(d):
    out_lines = ['"""Anh cau thu tu TheSportsDB v2 - khong sua tay."""\n\n', "PLAYER_PHOTOS_SPORTSDB = {\n"]
    for k, v in d.items():
        k_safe = k.replace('"', '\\"')
        out_lines.append(f'    "{k_safe}": "{v}",\n')
    out_lines.append("}\n")
    with open("player_photos_sportsdb.py", "w", encoding="utf-8") as f:
        f.writelines(out_lines)

def search_player_photo(full_name):
    """Tim anh cau thu, thu nhieu bien the ten."""
    variants = get_name_variants(full_name)
    
    for name in variants:
        try:
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={urllib.parse.quote(name)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; DUY-Analytics/2.0)"
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read())
                players = data.get("player", []) or []
                for p in players:
                    sport = (p.get("strSport") or "").lower()
                    if "soccer" in sport or "football" in sport:
                        thumb = p.get("strThumb") or p.get("strCutout") or ""
                        if thumb and thumb.startswith("http"):
                            return thumb, name  # (url, ten tim thay)
        except Exception:
            pass
        time.sleep(0.12)
    
    return None, None

# Chay
found = {}
not_found = []
total = len(PLAYERS)

print(f"Bat dau tim anh cho {total} cau thu (v2 - smart name matching)...")
print("-" * 60)

for i, player in enumerate(PLAYERS):
    photo_url, matched_name = search_player_photo(player)
    
    if photo_url:
        found[player] = photo_url
        if (i + 1) % 20 == 0 or i < 5:
            print(f"  [{i+1}/{total}] OK ({matched_name}): {player}")
    else:
        not_found.append(player)
    
    # Luu checkpoint moi 50 players
    if (i + 1) % 50 == 0:
        _save(found)
        print(f"  [Checkpoint {i+1}/{total}: {len(found)} found, {len(not_found)} not found]")

# Luu ket qua cuoi cung
_save(found)

print("\n" + "=" * 60)
print(f"HOAN TAT!")
print(f"  Tim duoc: {len(found)}/{total} anh ({100*len(found)//total}%)")
print(f"  Khong tim: {len(not_found)} cau thu")
print(f"  Luu vao: player_photos_sportsdb.py")
