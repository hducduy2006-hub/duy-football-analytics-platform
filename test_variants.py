"""Quick test of v2 name variant logic"""
import sys, json, time, urllib.request, urllib.parse, unicodedata
sys.stdout.reconfigure(encoding='utf-8')

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def get_name_variants(full_name):
    parts = full_name.split()
    variants = []
    if len(parts) <= 2:
        variants = [full_name, remove_accents(full_name)]
    elif len(parts) == 3:
        variants = [f"{parts[0]} {parts[1]}", f"{parts[0]} {parts[-1]}",
                    f"{parts[-2]} {parts[-1]}", parts[-1], full_name]
    else:
        variants = [f"{parts[0]} {parts[-2]}", f"{parts[0]} {parts[-1]}",
                    f"{parts[0]} {parts[1]}", parts[-2], parts[-1], full_name]
    
    expanded = []
    for v in variants:
        expanded.append(v)
        na = remove_accents(v)
        if na != v: expanded.append(na)
    
    seen = set()
    result = []
    for v in expanded:
        if v and v not in seen and len(v) > 2:
            seen.add(v); result.append(v)
    return result

# Test with hard StatsBomb names
test_cases = [
    "Lionel Andrés Messi Cuccittini",
    "Cristiano Ronaldo dos Santos Aveiro",
    "Kylian Mbappé Lottin",
    "Neymar da Silva Santos Junior",
    "Thomas Müller",
    "Aleksandr Golovin",
    "Ahmed Musa",
    "Luka Modrić",
]

for name in test_cases:
    variants = get_name_variants(name)
    print(f"\n{name}")
    print(f"  -> {variants[:4]}")  # Show top 4 variants
