"""
Tu dong sinh du lieu anh cau thu (PLAYER_PHOTOS) cho app DUY Analytics
=======================================================================

Muc dich: doc danh sach player_name THAT tu file worldcup_0_Extract.csv
(dung file ma app Streamlit cua ban dang doc), roi tra tung ten qua API
tim kiem cua SofaScore de lay id -> dung dung format URL anh ma ban dang
dung trong PLAYER_PHOTOS:
    https://img.res.sofascore.com/api/v1/player/{id}/image

Ket qua: file player_photos_generated.py chua 1 dict Python
    PLAYER_PHOTOS_GENERATED = { "Ten cau thu": "url anh", ... }

Cach dung trong app chinh (app.py) cua ban:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYER_PHOTOS.update(PLAYER_PHOTOS_GENERATED)   # dat ngay sau khi
    # khai bao PLAYER_PHOTOS goc, truoc khi dinh nghia get_player_avatar()

Cai dat truoc khi chay:
    pip install requests pandas

Chay (dung dung DATA_DIR ban dang dung trong app.py):
    python generate_player_photos.py "C:/Users/Admin/hdfs-practice/worldcup_0_Extract.csv"
"""

import sys
import time
import json

import pandas as pd
import requests

SOFASCORE_SEARCH_URL = "https://api.sofascore.com/api/v1/search/all"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"
}
REQUEST_DELAY_SEC = 0.6  # nghi giua cac request de tranh bi chan/rate-limit


def load_player_names(csv_path: str):
    df = pd.read_csv(csv_path, usecols=lambda c: c == "player_name", low_memory=False)
    names = sorted({str(n).strip() for n in df["player_name"].dropna().unique() if str(n).strip() and str(n) != "nan"})
    return names


def search_sofascore_player(name: str):
    """Tra 1 ten cau thu qua API tim kiem cua SofaScore, tra ve id neu tim thay."""
    params = {"q": name}
    try:
        resp = requests.get(SOFASCORE_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"  [Loi] '{name}': {exc}", file=sys.stderr)
        return None

    results = data.get("results", [])
    # Uu tien ket qua co type == "player" va entity co sport la football
    for r in results:
        if r.get("type") == "player":
            entity = r.get("entity", {})
            sport = entity.get("sport", {}).get("name", "")
            if sport.lower() == "football" or sport == "":
                return entity.get("id")
    return None


def main():
    if len(sys.argv) < 2:
        print("Cach dung: python generate_player_photos.py <duong_dan_toi_worldcup_0_Extract.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    names = load_player_names(csv_path)
    print(f"Tim thay {len(names)} ten cau thu duy nhat trong '{csv_path}'")

    photo_map = {}
    not_found = []

    for i, name in enumerate(names, start=1):
        print(f"[{i}/{len(names)}] Dang tra: {name} ...", end=" ")
        player_id = search_sofascore_player(name)
        if player_id:
            photo_map[name] = f"https://img.res.sofascore.com/api/v1/player/{player_id}/image"
            print(f"OK (id={player_id})")
        else:
            not_found.append(name)
            print("KHONG TIM THAY")
        time.sleep(REQUEST_DELAY_SEC)

    with open("player_photos_generated.py", "w", encoding="utf-8") as f:
        f.write('"""File duoc sinh tu dong boi generate_player_photos.py - KHONG sua tay."""\n\n')
        f.write("PLAYER_PHOTOS_GENERATED = ")
        f.write(json.dumps(photo_map, indent=4, ensure_ascii=False))
        f.write("\n")

    print(f"\nDa luu {len(photo_map)} anh vao 'player_photos_generated.py'")
    if not_found:
        print(f"Khong tim thay anh cho {len(not_found)} cau thu (se dung fallback avatar chu cai):")
        for n in not_found:
            print(f"  - {n}")


if __name__ == "__main__":
    main()