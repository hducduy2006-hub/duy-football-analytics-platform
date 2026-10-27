import base64, sys
sys.stdout.reconfigure(encoding='utf-8')

def _make_jersey_svg(initials, primary_color, secondary_color):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">'
        f'<circle cx="40" cy="40" r="38" fill="{primary_color}"/>'
        f'<text x="40" y="50" font-size="20" font-weight="900" text-anchor="middle"'
        f' fill="{secondary_color}" font-family="Arial Black,Arial,sans-serif">{initials}</text>'
        f'</svg>'
    )
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()

tests = [
    ('Germany', '#000000', '#FFFFFF', 'KM'),
    ('Brazil', '#009C3B', '#FFDF00', 'NJ'),
    ('France', '#003399', '#FFFFFF', 'KM'),
]

for team, pri, sec, init in tests:
    result = _make_jersey_svg(init, pri, sec)
    print(f'{team}: OK - length={len(result)}')

print('SVG generation working!')
