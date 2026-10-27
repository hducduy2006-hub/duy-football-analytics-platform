"""
Script tai anh cau thu tu Wikipedia ve may local.
Chay 1 lan: python download_player_photos.py
Anh se duoc luu vao thu muc: player_photos/
"""
import os, sys, time, json, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')


try:
    from player_photos_generated import PLAYER_PHOTOS_GENERATED
except ImportError:
    print("❌ Không tìm thấy player_photos_generated.py")
    exit(1)

SAVE_DIR = "player_photos"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Referer": "https://en.wikipedia.org/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
}

saved_map = {}
failed = []

total = len(PLAYER_PHOTOS_GENERATED)
print(f"🚀 Bắt đầu tải {total} ảnh cầu thủ...")

for i, (player_name, url) in enumerate(PLAYER_PHOTOS_GENERATED.items()):
    # Tạo tên file an toàn
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in player_name).strip()
    safe_name = safe_name[:60]
    
    # Xác định extension
    ext = ".jpg"
    if ".png" in url.lower(): ext = ".png"
    elif ".webp" in url.lower(): ext = ".webp"
    elif ".gif" in url.lower(): ext = ".gif"
    
    fname = os.path.join(SAVE_DIR, f"{safe_name}{ext}")
    
    # Skip nếu đã tải
    if os.path.exists(fname) and os.path.getsize(fname) > 2000:
        saved_map[player_name] = fname
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{total}] Skip (đã có): {player_name}")
        continue
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            if len(data) > 2000:  # File hợp lệ (> 2KB)
                with open(fname, "wb") as f:
                    f.write(data)
                saved_map[player_name] = fname
                if (i+1) % 20 == 0 or i < 5:
                    print(f"  [{i+1}/{total}] ✅ {player_name}")
            else:
                failed.append(player_name)
                if (i+1) % 50 == 0:
                    print(f"  [{i+1}/{total}] ⚠️  File quá nhỏ: {player_name}")
    except Exception as e:
        failed.append(player_name)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{total}] ❌ Lỗi: {player_name}: {e}")
    
    # Dừng nhẹ để tránh bị block
    time.sleep(0.05)

# Lưu index ánh xạ tên -> đường dẫn file local
index_path = os.path.join(SAVE_DIR, "photo_index.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(saved_map, f, ensure_ascii=False, indent=2)

print(f"\n✅ Hoàn tất!")
print(f"   Đã tải: {len(saved_map)} ảnh -> thư mục '{SAVE_DIR}/'")
print(f"   Thất bại: {len(failed)} cầu thủ (sẽ dùng avatar chữ cái)")
print(f"   Index: {index_path}")
