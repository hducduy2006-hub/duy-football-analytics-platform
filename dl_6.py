import os, sys, io, re, json, unicodedata, requests
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
OUT = r"C:\Users\Admin\Downloads\spark-docker\spark-docker\player_photos"
UA = {"User-Agent": "DUYFootballAnalytics/1.0 (https://github.com/duy-analytics; hoangducduy.student@gmail.com) python-requests/2"}
sess = requests.Session()
sess.headers.update(UA)

def slug(n):
    n = "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", n).strip("_")

targets = {
    "Ghilane Chalali": "https://upload.wikimedia.org/wikipedia/commons/7/7e/PAN-TUN_%2818%29_%28cropped%29.jpg",
    "Ramadan Sobhi": "https://r2.thesportsdb.com/images/media/player/thumb/0k6mey1509655925.jpg",
    "Manuel Marouan da Costa Trinidade": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Spain_vs_Morocco_%2835%29_%28cropped%29.jpg",
    "Mouez Hassen": "https://r2.thesportsdb.com/images/media/player/thumb/8qt3om1551798727.jpg",
    "Mahmoud Abdul Monem Abdelhamid Soliman": "https://thumb.wikimedia.org/wikipedia/commons/thumb/7/7b/Mahmoud_Kahraba_2018_%28cropped%29.jpg/330px-Mahmoud_Kahraba_2018_%28cropped%29.jpg",
    "Mohanad Aseri Abu Radiah": "https://thumb.wikimedia.org/wikipedia/commons/thumb/3/34/Muhannad_Assiri.jpg/330px-Muhannad_Assiri.jpg",
}

idx_path = os.path.join(OUT, "photo_index.json")
index = json.load(open(idx_path, encoding="utf-8"))

for name, url in targets.items():
    try:
        r = sess.get(url, timeout=25)
        if r.status_code == 200 and len(r.content) > 1000:
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            w, h = im.size
            s = min(w, h)
            l = (w - s) // 2
            t = 0 if h > w else (h - s) // 2
            fname = slug(name) + ".jpg"
            im.crop((l, t, l + s, t + s)).resize((200, 200), Image.LANCZOS).save(
                os.path.join(OUT, fname), "JPEG", quality=85
            )
            index[name] = fname
            print(f"SUCCESS: {name} -> {fname}")
        else:
            print(f"FAILED (status {r.status_code}): {name}")
    except Exception as e:
        print(f"ERROR: {name} -> {e}")

json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Current total in index: {len(index)} / 600")
