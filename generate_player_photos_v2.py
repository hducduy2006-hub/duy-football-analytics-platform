"""
Tu dong sinh du lieu anh cau thu (PLAYER_PHOTOS) cho app DUY Analytics
=======================================================================
PHIEN BAN 2 - khong dung SofaScore nua (bi chan 403 / Cloudflare).

Nguon anh moi: file CompleteDataset.csv trong repo GitHub cong khai
khanhnamle1994/world-cup-2018 (du lieu cau thu FIFA 18, anh tinh luu
tren cdn.sofifa.org - khong phai API tim kiem nen KHONG bi chan nhu
SofaScore search API).

Van de: ten trong du lieu WC 2018 cua ban (StatsBomb) la ten day du,
vi du "Lionel Andres Messi Cuccittini", con ten trong file FIFA18 lai
la ten ngan/biet danh, vi du "L. Messi". Script nay dung thuat toan
khop token co trong so IDF (uu tien tu hiem gap nhu "Messi", "Neymar"
thay vi bi lech boi ho pho bien nhu "da", "silva", "santos") de tim
ban ghi khop nhat.

Ket qua: file player_photos_generated.py chua dict
    PLAYER_PHOTOS_GENERATED = { "Ten cau thu": "url anh", ... }
kem theo file match_report.csv de ban KIEM TRA LAI cac truong hop
diem tin cay thap (thuat toan khop ten khong bao gio dung 100%).

Cach dung trong app.py:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
    PLAYER_PHOTOS.update(PLAYER_PHOTOS_GENERATED)   # sau khi khai bao
    # PLAYER_PHOTOS goc, truoc khi dinh nghia get_player_avatar()

Cai dat truoc khi chay:
    pip install requests pandas

Chay:
    python generate_player_photos_v2.py "C:/Users/Admin/hdfs-practice/worldcup_0_Extract.csv"
"""

import sys
import os
import math
import re
import json
import unicodedata
from collections import defaultdict

import requests
import pandas as pd

DATASET_URL = "https://raw.githubusercontent.com/khanhnamle1994/world-cup-2018/master/CompleteDataset.csv"
DATASET_CACHE = "fifa18_complete_dataset.csv"
LOW_CONFIDENCE_THRESHOLD = 7.5  # duoi nguong nay -> can nguoi kiem tra lai


def download_dataset_if_needed():
    if os.path.exists(DATASET_CACHE):
        print(f"Da co san '{DATASET_CACHE}', bo qua tai lai.")
        return
    print("Dang tai du lieu anh cau thu tu GitHub (mot lan duy nhat)...")
    resp = requests.get(DATASET_URL, timeout=120)
    resp.raise_for_status()
    with open(DATASET_CACHE, "wb") as f:
        f.write(resp.content)
    print(f"Da tai xong: {len(resp.content) / 1_000_000:.1f} MB")


def normalize_tokens(name: str):
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z\s]", " ", name).lower()
    return set(t for t in name.split() if len(t) > 1)


def build_index(df: pd.DataFrame):
    inverted = defaultdict(list)
    for idx, name in df["Name"].items():
        for tok in normalize_tokens(name):
            inverted[tok].append(idx)
    freq = {tok: len(idxs) for tok, idxs in inverted.items()}
    n_total = len(df)
    idf_cache = {}

    def idf(tok):
        if tok not in idf_cache:
            idf_cache[tok] = math.log(n_total / (1 + freq.get(tok, 0)))
        return idf_cache[tok]

    return inverted, idf


def find_best_match(query_name: str, df: pd.DataFrame, inverted, idf):
    q_tokens = normalize_tokens(query_name)
    candidate_shared = defaultdict(set)
    for tok in q_tokens:
        for idx in inverted.get(tok, []):
            candidate_shared[idx].add(tok)

    if not candidate_shared:
        return None

    def score(idx):
        shared = candidate_shared[idx]
        max_idf = max(idf(t) for t in shared)
        sum_idf = sum(idf(t) for t in shared)
        overall = df.loc[idx, "Overall"]
        return (round(max_idf, 4), round(sum_idf, 4), overall)

    best_idx = max(candidate_shared.keys(), key=score)
    confidence = score(best_idx)[0]  # max_idf la chi so tin cay chinh
    return {
        "matched_name": df.loc[best_idx, "Name"],
        "photo_url": df.loc[best_idx, "Photo"],
        "confidence": round(confidence, 2),
    }


def load_target_player_names(csv_path: str):
    df = pd.read_csv(csv_path, usecols=lambda c: c == "player_name", low_memory=False)
    return sorted({str(n).strip() for n in df["player_name"].dropna().unique() if str(n).strip() and str(n) != "nan"})


def main():
    if len(sys.argv) < 2:
        print("Cach dung: python generate_player_photos_v2.py <duong_dan_toi_worldcup_0_Extract.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    download_dataset_if_needed()

    print("Dang nap du lieu FIFA18 vao bo nho...")
    fifa_df = pd.read_csv(DATASET_CACHE, usecols=["Name", "Photo", "Overall"], low_memory=False)
    fifa_df = fifa_df.dropna(subset=["Name", "Photo"]).reset_index(drop=True)
    print(f"Da nap {len(fifa_df)} cau thu FIFA18 lam nguon tra cuu anh")

    print("Dang dung index tra cuu...")
    inverted, idf = build_index(fifa_df)

    names = load_target_player_names(csv_path)
    print(f"Tim thay {len(names)} ten cau thu can tra anh trong '{csv_path}'")

    photo_map = {}
    report_rows = []
    low_confidence_count = 0

    for i, name in enumerate(names, start=1):
        result = find_best_match(name, fifa_df, inverted, idf)
        if result:
            photo_map[name] = result["photo_url"]
            flag = ""
            if result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                flag = "KIEM TRA LAI"
                low_confidence_count += 1
            report_rows.append({
                "ten_goc": name,
                "khop_voi": result["matched_name"],
                "do_tin_cay": result["confidence"],
                "ghi_chu": flag,
                "photo_url": result["photo_url"],
            })
            print(f"[{i}/{len(names)}] {name} -> {result['matched_name']} (tin cay: {result['confidence']}) {flag}")
        else:
            report_rows.append({
                "ten_goc": name, "khop_voi": "", "do_tin_cay": 0,
                "ghi_chu": "KHONG TIM THAY", "photo_url": "",
            })
            print(f"[{i}/{len(names)}] {name} -> KHONG TIM THAY")

    with open("player_photos_generated.py", "w", encoding="utf-8") as f:
        f.write('"""File duoc sinh tu dong boi generate_player_photos_v2.py - KHONG sua tay."""\n\n')
        f.write("PLAYER_PHOTOS_GENERATED = ")
        f.write(json.dumps(photo_map, indent=4, ensure_ascii=False))
        f.write("\n")

    pd.DataFrame(report_rows).to_csv("match_report.csv", index=False, encoding="utf-8-sig")

    print(f"\nDa luu {len(photo_map)} anh vao 'player_photos_generated.py'")
    print(f"Da luu bao cao doi chieu vao 'match_report.csv'")
    print(f"Co {low_confidence_count} truong hop do tin cay thap can ban kiem tra lai bang mat"
          f" (mo match_report.csv, loc cot 'ghi_chu' = 'KIEM TRA LAI')")


if __name__ == "__main__":
    main()