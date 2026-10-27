"""Download photos for players whose StatsBomb full name doesn't match Wikipedia (manual title map)."""
import sys, os, re, io, json, time, unicodedata, requests
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\Users\Admin\Downloads\spark-docker\spark-docker\player_photos"
UA = {"User-Agent": "DUYFootballAnalytics/1.0 (https://github.com/duy-analytics; hoangducduy.student@gmail.com) python-requests/2"}

TITLES = {
    "Anderson Santamaría Bardales": "Anderson Santamaría",
    "Ghilane Chalali": "Ghilane Chalali",
    "Aníbal Cesis Godoy": "Aníbal Godoy",
    "Ramadan Sobhi": "Ramadan Sobhi",
    "Hussain Al Mogahwi": "Hussain Al-Moqahwi",
    "Osama Abdulrezag Hawsawi": "Osama Hawsawi",
    "Manuel Marouan da Costa Trinidade": "Manuel da Costa",
    "Adrien Sebastian Perruchet Silva": "Adrien Silva",
    'M"Baye Babacar Niang': "M'Baye Niang",
    "Mouez Hassen": "Mouez Hassen",
    "Mahmoud Abdul Monem Abdelhamid Soliman": "Kahraba",
    "Mohanad Aseri Abu Radiah": "Mohannad Asiri",
    "Ian Rey Smith Quiros": "Ian Smith (Costa Rican footballer)",
}

def slug(n):
    n = "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", n).strip("_")

idx_path = os.path.join(OUT, "photo_index.json")
index = json.load(open(idx_path, encoding="utf-8"))
for name, title in TITLES.items():
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, params={
        "action": "query", "format": "json", "titles": title, "redirects": 1,
        "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 330}, timeout=20).json()
    page = next(iter(r["query"]["pages"].values()))
    src = page.get("thumbnail", {}).get("source")
    if not src:
        print("NO IMAGE:", name, "->", page.get("title")); continue
    img = requests.get(src, headers=UA, timeout=25)
    if img.status_code != 200:
        print("HTTP", img.status_code, name); continue
    im = Image.open(io.BytesIO(img.content)).convert("RGB")
    w, h = im.size; s = min(w, h); l = (w - s) // 2; t = 0 if h > w else (h - s) // 2
    im.crop((l, t, l + s, t + s)).resize((200, 200), Image.LANCZOS).save(os.path.join(OUT, slug(name) + ".jpg"), "JPEG", quality=85)
    index[name] = slug(name) + ".jpg"
    print("OK:", name, "->", page.get("title"))
    time.sleep(0.5)
json.dump(index, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("TOTAL in index:", len(index))
