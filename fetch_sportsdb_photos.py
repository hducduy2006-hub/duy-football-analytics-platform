"""
Lay anh cau thu tu TheSportsDB API (mien phi, khong bi chan).
Luu vao player_photos_sportsdb.py de import vao app.py
"""
import sys, json, time, urllib.request, urllib.parse, os
sys.stdout.reconfigure(encoding='utf-8')

try:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYERS = list(PLAYER_PHOTOS_GENERATED.keys())
except ImportError:
    print("Khong tim thay player_photos_generated.py")
    exit(1)

def search_player_photo(full_name):
    """Tim anh cau thu tren TheSportsDB theo ten day du hoac ten viet tat."""
    names_to_try = []
    parts = full_name.split()
    
    # Thu cac bien the ten
    names_to_try.append(full_name)                          # Ten day du
    if len(parts) >= 2:
        names_to_try.append(f"{parts[0]} {parts[-1]}")     # Ten dau + Ho cuoi
        names_to_try.append(parts[-1])                     # Chi ho
        names_to_try.append(f"{parts[-1]} {parts[0]}")    # Ho truoc
    
    for name in names_to_try:
        try:
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={urllib.parse.quote(name)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DUY-Analytics/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read())
                players = data.get("player", []) or []
                for p in players:
                    sport = (p.get("strSport") or "").lower()
                    if "soccer" in sport or "football" in sport:
                        thumb = p.get("strThumb") or p.get("strCutout") or ""
                        if thumb and thumb.startswith("http"):
                            return thumb
        except Exception:
            pass
        time.sleep(0.15)
    return None

# Ket qua
found = {}
not_found = []
total = len(PLAYERS)
save_every = 50  # Luu file trung gian moi 50 cau thu

print(f"Bat dau tim anh cho {total} cau thu tu TheSportsDB...")
print("(Co the mat 5-8 phut)")
print("-" * 50)

for i, player in enumerate(PLAYERS):
    photo_url = search_player_photo(player)
    
    if photo_url:
        found[player] = photo_url
        if (i + 1) % 20 == 0 or i < 3:
            print(f"  [{i+1}/{total}] OK: {player}")
    else:
        not_found.append(player)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] NOT FOUND: {player}")
    
    # Luu file trung gian
    if (i + 1) % save_every == 0:
        out = "PLAYER_PHOTOS_SPORTSDB = {\n"
        for k, v in found.items():
            k_safe = k.replace('"', '\\"')
            out += f'    "{k_safe}": "{v}",\n'
        out += "}\n"
        with open("player_photos_sportsdb.py", "w", encoding="utf-8") as f:
            f.write('"""Anh cau thu tu TheSportsDB - khong sua tay."""\n\n')
            f.write(out)
        print(f"  [Saved checkpoint: {len(found)} players found so far]")

# Luu file cuoi cung
out = "PLAYER_PHOTOS_SPORTSDB = {\n"
for k, v in found.items():
    k_safe = k.replace('"', '\\"')
    out += f'    "{k_safe}": "{v}",\n'
out += "}\n"

with open("player_photos_sportsdb.py", "w", encoding="utf-8") as f:
    f.write('"""Anh cau thu tu TheSportsDB - khong sua tay."""\n\n')
    f.write(out)

print("\n" + "=" * 50)
print(f"XONG! Tim duoc: {len(found)}/{total} anh")
print(f"Khong tim thay: {len(not_found)} cau thu")
print(f"Da luu vao: player_photos_sportsdb.py")
if not_found:
    print(f"\nCau thu khong tim thay ({len(not_found)}):")
    for p in not_found[:20]:
        print(f"  - {p}")
