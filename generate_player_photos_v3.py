"""
Tu dong sinh du lieu anh cau thu (PLAYER_PHOTOS) cho app DUY Analytics
=======================================================================
PHIEN BAN 3 - dung Wikipedia thay vi SofaScore/SoFIFA

Ly do doi nguon:
  - SofaScore (v1): bi chan 403 (Cloudflare bot protection) -> 0% thanh cong
  - SoFIFA/cdn.sofifa.org (v2): domain da bi khai tu (DNS_PROBE_FINISHED_
    NXDOMAIN), du lieu tu 2017 nen link chet theo thoi gian
  - Wikipedia (v3): API cong khai, on dinh hon 10 nam, khong chan bot voi
    tan suat request vua phai + User-Agent hop le. Anh lay tu chinh anh
    infobox cua bai Wikipedia cau thu do (thuong la anh that, co giay
    phep ro rang tren Wikimedia Commons).

Cach hoat dong: voi moi ten cau thu, goi 1 API call toi Wikipedia
(generator=search + prop=pageimages) de vua tim bai viet gan dung nhat
vua lay luon anh dai dien - chi 1 request/cau thu, khong can 2 buoc
nhu Wikidata.

Ket qua:
  - player_photos_generated.py  -> dict PLAYER_PHOTOS_GENERATED
  - match_report.csv            -> bang doi chieu de kiem tra lai

Cach dung trong app.py (giu nguyen nhu cac ban truoc):
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYER_PHOTOS.update(PLAYER_PHOTOS_GENERATED)

Cai dat truoc khi chay:
    pip install requests pandas

Chay:
    python generate_player_photos_v3.py "C:/Users/Admin/hdfs-practice/worldcup_0_Extract.csv"
"""

import sys
import time
import json

import requests
import pandas as pd

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia yeu cau User-Agent mo ta ro rang (khong phai bot spam) -
# neu khong co the bi gioi han toc do manh hon.
HEADERS = {
    "User-Agent": "DUYAnalyticsPlayerPhotoTool/1.0 (personal football analytics project; contact: n/a)"
}

REQUEST_DELAY_SEC = 1.2  # Wikipedia khuyen nghi khong qua ~1 request/giay
MAX_RETRIES = 5
THUMB_SIZE = 300


def fetch_wikipedia_thumbnail(name: str):
    """
    Dung generator=search de tim bai viet Wikipedia gan dung nhat voi
    ten cau thu, roi lay luon anh thumbnail (pageimages) cua bai do.
    Tra ve (matched_title, image_url) hoac None neu khong co.
    Tu dong cho va thu lai neu bi loi 429 (goi qua nhanh).
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{name} footballer",
        "gsrlimit": 1,
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": THUMB_SIZE,
    }

    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait_s = int(resp.headers.get("Retry-After", 5)) * attempt
                print(f"  [429] '{name}': bi gioi han toc do, cho {wait_s}s roi thu lai "
                      f"(lan {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            print(f"  [Loi] '{name}' (lan {attempt}/{MAX_RETRIES}): {exc}", file=sys.stderr)
            time.sleep(2 * attempt)

    if data is None:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    # Chi co 1 ket qua (gsrlimit=1)
    page = next(iter(pages.values()))
    title = page.get("title", "")
    thumbnail = page.get("thumbnail", {})
    image_url = thumbnail.get("source")

    if not image_url:
        return None
    return title, image_url


def load_target_player_names(csv_path: str):
    df = pd.read_csv(csv_path, usecols=lambda c: c == "player_name", low_memory=False)
    return sorted({str(n).strip() for n in df["player_name"].dropna().unique() if str(n).strip() and str(n) != "nan"})


def main():
    if len(sys.argv) < 2:
        print("Cach dung: python generate_player_photos_v3.py <duong_dan_toi_worldcup_0_Extract.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    names = load_target_player_names(csv_path)
    print(f"Tim thay {len(names)} ten cau thu can tra anh trong '{csv_path}'")

    photo_map = {}
    report_rows = []
    not_found = []

    for i, name in enumerate(names, start=1):
        result = fetch_wikipedia_thumbnail(name)
        if result:
            matched_title, image_url = result
            photo_map[name] = image_url
            report_rows.append({
                "ten_goc": name, "khop_voi_bai_wiki": matched_title,
                "photo_url": image_url, "ghi_chu": "",
            })
            print(f"[{i}/{len(names)}] {name} -> {matched_title}")
        else:
            not_found.append(name)
            report_rows.append({
                "ten_goc": name, "khop_voi_bai_wiki": "",
                "photo_url": "", "ghi_chu": "KHONG TIM THAY",
            })
            print(f"[{i}/{len(names)}] {name} -> KHONG TIM THAY")
        time.sleep(REQUEST_DELAY_SEC)

    with open("player_photos_generated.py", "w", encoding="utf-8") as f:
        f.write('"""File duoc sinh tu dong boi generate_player_photos_v3.py - KHONG sua tay."""\n\n')
        f.write("PLAYER_PHOTOS_GENERATED = ")
        f.write(json.dumps(photo_map, indent=4, ensure_ascii=False))
        f.write("\n")

    pd.DataFrame(report_rows).to_csv("match_report.csv", index=False, encoding="utf-8-sig")

    print(f"\nDa luu {len(photo_map)} anh vao 'player_photos_generated.py'")
    print(f"Da luu bao cao doi chieu vao 'match_report.csv'")
    if not_found:
        print(f"Khong tim thay anh cho {len(not_found)} cau thu (van dung fallback avatar chu cai binh thuong)")


if __name__ == "__main__":
    main()