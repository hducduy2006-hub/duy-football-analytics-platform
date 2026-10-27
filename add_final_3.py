import os, json, sys
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
OUT = r"C:\Users\Admin\Downloads\spark-docker\spark-docker\player_photos"
idx_path = os.path.join(OUT, "photo_index.json")
index = json.load(open(idx_path, encoding="utf-8"))

src_map = {
    "Anderson Santamaría Bardales": (
        r"C:\Users\Admin\.gemini\antigravity\brain\bec7a45c-4651-4c0f-95b1-4af58b62a2ff\anderson_santamaria_bardales_1791863587531.jpg",
        "anderson_santamaria_bardales.jpg"
    ),
    "Hussain Al Mogahwi": (
        r"C:\Users\Admin\.gemini\antigravity\brain\bec7a45c-4651-4c0f-95b1-4af58b62a2ff\hussain_al_mogahwi_1791863604860.jpg",
        "hussain_al_mogahwi.jpg"
    ),
    "Ian Rey Smith Quiros": (
        r"C:\Users\Admin\.gemini\antigravity\brain\bec7a45c-4651-4c0f-95b1-4af58b62a2ff\ian_rey_smith_quiros_1791863623195.jpg",
        "ian_rey_smith_quiros.jpg"
    ),
}

for name, (src, dest_name) in src_map.items():
    if os.path.exists(src):
        im = Image.open(src).convert("RGB")
        w, h = im.size
        s = min(w, h)
        l = (w - s) // 2
        t = 0 if h > w else (h - s) // 2
        im_crop = im.crop((l, t, l + s, t + s)).resize((200, 200), Image.LANCZOS)
        dest_path = os.path.join(OUT, dest_name)
        im_crop.save(dest_path, "JPEG", quality=90)
        index[name] = dest_name
        print(f"Added: {name} -> {dest_name}")
    else:
        print(f"Source not found: {src}")

json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"TOTAL in index now: {len(index)} / 600")
