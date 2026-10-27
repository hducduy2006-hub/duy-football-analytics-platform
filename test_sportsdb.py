"""
Test TheSportsDB API - free public API with real player headshots
"""
import sys, json, urllib.request, time
sys.stdout.reconfigure(encoding='utf-8')

def search_player(name):
    url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={urllib.parse.quote(name)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
            players = data.get("player", []) or []
            # Filter soccer/football
            for p in players:
                sport = (p.get("strSport") or "").lower()
                if "soccer" in sport or "football" in sport:
                    thumb = p.get("strThumb") or p.get("strCutout") or ""
                    if thumb:
                        return thumb
    except Exception as e:
        print(f"  Error: {e}")
    return None

import urllib.parse

test_players = [
    "Kylian Mbappe",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Neymar",
    "Harry Kane",
    "Luka Modric",
    "Antoine Griezmann",
    "Eden Hazard",
]

print("Testing TheSportsDB API...")
for p in test_players:
    url = search_player(p)
    status = "FOUND" if url else "NOT FOUND"
    print(f"  {p}: {status}")
    if url:
        print(f"    -> {url[:80]}...")
    time.sleep(0.3)
