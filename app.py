import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import json
import base64
import hashlib
import html
import unicodedata as _ud
import urllib.request
from pro_features import (
    tr,
    load_opta_ucl_match_bundle,
    render_sidebar_pro_controls,
    render_player_radar_and_fifa_subtab,
    render_match_action_bar,
    render_attack_momentum_and_defense,
    render_match_ai_timeline_news_tab,
    render_live_and_multimatch_workspace,
    render_tournament_hub_workspace,
    render_compare_ai_and_architecture_workspace,
)

# --- CẤU HÌNH GIAO DIỆN CHUẨN SOFASCORE / FOTMOB ---
st.set_page_config(
    page_title="SofaScore - Football Analytics & Big Data Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* SofaScore Dark Theme */
    .stApp {
        background-color: #0E121A;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* SofaScore Match Header Banner */
    .match-header-card {
        background: linear-gradient(135deg, #131A26 0%, #1A2436 100%);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        border: 1px solid #283750;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    }
    .match-top-meta {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        align-items: center;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.14);
    }
    .score-badge {
        background: radial-gradient(circle at 50% 35%, #172236 0%, #0B0F19 100%);
        padding: 14px 28px;
        min-width: 176px;
        border-radius: 16px;
        border: 1.5px solid #38BDF8;
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.22);
    }
    .score-numbers {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        justify-items: center;
        gap: 14px;
        width: 100%;
        margin: 0;
        font-size: 44px;
        line-height: 1;
        color: #FFFFFF;
        font-weight: 900;
        font-variant-numeric: tabular-nums;
    }
    .score-digit.home { justify-self: end; text-align: right; }
    .score-digit.away { justify-self: start; text-align: left; }
    .score-sep { color: #64748B; font-weight: 800; font-size: 36px; line-height: 1; }
    .pen-badge {
        background: rgba(250, 204, 21, 0.16);
        border: 1px solid #FACC15;
        color: #FDE047;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.08em;
        padding: 3px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        display: inline-block;
        margin-top: 8px;
    }
    .match-score-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        align-items: center;
        gap: 28px;
        margin: 22px 0 20px;
    }
    .match-team-name {
        min-width: 0;
        margin: 0 !important;
        padding: 0 !important;
        font-size: clamp(22px, 2.4vw, 32px);
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.25;
        overflow-wrap: anywhere;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .match-team-name.home { justify-content: flex-end; text-align: right; }
    .match-team-name.away { justify-content: flex-start; text-align: left; }
    .match-team-col {
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-width: 0;
    }
    .match-team-col.home { align-items: flex-end; }
    .match-team-col.away { align-items: flex-start; }
    .team-meta-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
    }
    .team-meta-pill {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid #2A3B54;
        color: #CBD5E1;
        font-size: 11.5px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        white-space: nowrap;
    }
    .team-meta-pill.home-accent {
        border-color: rgba(56, 189, 248, 0.45);
        color: #7DD3FC;
        background: rgba(56, 189, 248, 0.12);
    }
    .team-meta-pill.away-accent {
        border-color: rgba(255, 77, 109, 0.45);
        color: #FDA4AF;
        background: rgba(255, 77, 109, 0.12);
    }
    .match-goal-lists {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 16px;
        margin: 0 0 16px;
    }
    .match-goal-side {
        min-width: 0;
        padding: 12px 16px;
        border: 1px solid #29364b;
        border-radius: 10px;
        background: rgba(8, 13, 22, 0.6);
    }
    .match-goal-side.home { border-left: 3.5px solid #38BDF8; }
    .match-goal-side.away { border-left: 3.5px solid #FF4D6D; }
    .match-goal-team {
        margin-bottom: 8px;
        color: #E2E8F0;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .04em;
        overflow-wrap: anywhere;
    }
    .match-goal-rows { display: grid; gap: 6px; }
    .match-goal-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        min-width: 0;
        color: #F1F5F9;
        font-size: 14px;
        line-height: 1.35;
    }
    .match-goal-minute {
        flex: 0 0 48px;
        color: #9FB0C8;
        font-weight: 700;
        white-space: nowrap;
    }
    .match-goal-player { min-width: 0; overflow-wrap: anywhere; }
    .match-goal-empty { color: #94A3B8; font-size: 13px; }

    /* FotMob / SofaScore Pro Pill Tabs */
    [data-baseweb="tab-list"] {
        background: #121926 !important;
        border: 1px solid #26354D !important;
        border-radius: 14px !important;
        padding: 6px 8px !important;
        gap: 8px !important;
        margin-bottom: 18px !important;
    }
    [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 18px !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        color: #94A3B8 !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.18s ease !important;
    }
    [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(56, 189, 248, 0.10) !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.38) !important;
    }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
        display: none !important;
    }

    @media (max-width: 640px) {
        .match-header-card { padding: 18px 14px; }
        .match-top-meta { grid-template-columns: 1fr; text-align: center; gap: 6px; }
        .match-score-row { gap: 10px; }
        .match-team-name { font-size: clamp(17px, 4.8vw, 22px); }
        .score-badge { min-width: 124px; padding: 10px 14px; }
        .score-numbers { font-size: 30px; gap: 8px; }
        .match-goal-lists { grid-template-columns: minmax(0, 1fr); gap: 10px; }
    }

    /* Player Cards & Sidebar Styling */
    .player-card {
        background-color: #141B27;
        border-radius: 12px;
        padding: 14px;
        border: 1px solid #232E42;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 12px;
    }
    .player-card:hover {
        transform: translateY(-3px);
        border-color: #2C66FF;
    }

    /* SofaScore Player Match Detail Modal/Card */
    .sofa-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 4px;
    }
    .sofa-impact-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px 16px;
    }
    .sofa-gk-strip {
        grid-column: 1 / -1;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-top: 4px;
    }
    @media (max-width: 720px) {
        .sofa-stat-grid, .sofa-impact-grid { grid-template-columns: minmax(0, 1fr); }
        .sofa-gk-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .sofa-stat-box {
        background: #0E131E;
        border-radius: 11px;
        padding: 10px 12px;
        border: 1px solid #1C2536;
        margin: 0;
    }
    .sofa-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding: 4.5px 0;
        border-bottom: 1px solid #18202E;
        font-size: 12.5px;
    }
    .sofa-stat-row:last-child {
        border-bottom: none;
        padding-bottom: 1px;
    }
    .stat-name {
        color: #94A3B8;
        font-weight: 500;
        line-height: 1.25;
    }
    .stat-val {
        color: #FFFFFF;
        font-weight: 800;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    /* Dashboard polish: match the pitch and chart surfaces */
    :root {
        color-scheme: dark;
        --dashboard-bg: #0D100E;
        --dashboard-surface: #131A26;
        --dashboard-border: #273448;
        --dashboard-muted: #9AA9BE;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--dashboard-bg);
    }
    [data-testid="stMainBlockContainer"], .main .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none;
    }
    [data-testid="stHeader"] button,
    [data-testid="stToolbar"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        pointer-events: auto;
    }
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* Consistent KPI cards */
    [data-testid="stMetric"] {
        min-height: 98px;
        padding: 14px 16px !important;
        border: 1px solid var(--dashboard-border) !important;
        border-radius: 14px !important;
        background: linear-gradient(145deg, #172131 0%, var(--dashboard-surface) 68%, #111711 100%);
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.20);
        transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #3567B8 !important;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--dashboard-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    [data-testid="stMetricValue"] {
        color: #F4F7FB !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* Rounded, layered surfaces for bordered containers and charts */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--dashboard-border) !important;
        border-radius: 16px !important;
        background: linear-gradient(150deg, rgba(19, 26, 38, 0.94), rgba(13, 16, 14, 0.96));
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: inherit;
    }
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
    }

    /* Compact, readable global filters & Wide Non-Clipping Match Selector */
    [data-testid="stSidebar"] {
        background: #10151E;
        border-right: 1px solid #202B3B;
    }
    @media (min-width: 992px) {
        [data-testid="stSidebar"] {
            min-width: 320px !important;
            width: 320px !important;
        }
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #AAB7C9 !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        margin: 0.12rem 0 0.48rem;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        font-size: 12.8px !important;
        padding-left: 9px !important;
        padding-right: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] [role="combobox"] {
        border-radius: 10px !important;
        border-color: #2A374B !important;
        background-color: #131A26 !important;
    }
    /* Mở rộng menu xổ xuống để hiển thị trọn vẹn Tên 2 Đội + Tỷ Số + Ngày */
    div[data-baseweb="popover"]:has(ul[role="listbox"]),
    div[data-baseweb="popover"]:has([data-testid="stVirtualDropdown"]) {
        min-width: 375px !important;
        max-width: min(500px, 94vw) !important;
    }
    ul[role="listbox"] li[role="option"],
    [data-testid="stVirtualDropdown"] li[role="option"] {
        font-size: 13px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #263348;
        border-radius: 12px;
        background: rgba(19, 26, 38, 0.62);
    }
    .sidebar-filter-title {
        margin: 0 0 0.65rem;
        padding: 0 0 0.7rem;
        border-bottom: 1px solid #263348;
        color: #DCE6F5;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-filter-subtitle {
        margin: 0.5rem 0 0.1rem;
        color: #8798B0;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    /* Tighten dashboard spacing on narrow screens */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"], .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        [data-testid="stMetric"] {
            min-height: 86px;
            padding: 11px 12px !important;
        }
    }

</style>
""", unsafe_allow_html=True)


DATA_DIR = "C:/Users/Admin/hdfs-practice"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "."

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)


# --- MÀU SẮC ĐỘI BÓNG (QUỐC GIA & CLB) ---
_TEAM_COLORS = {
    # Đội tuyển
    "Russia":          ("#CC0000", "#003399"),
    "Saudi Arabia":    ("#006C35", "#FFFFFF"),
    "Egypt":           ("#CE1126", "#FFFFFF"),
    "Uruguay":         ("#75AADB", "#FFFFFF"),
    "Morocco":         ("#C1272D", "#006233"),
    "Iran":            ("#239F40", "#FFFFFF"),
    "Portugal":        ("#006600", "#FF0000"),
    "Spain":           ("#C60B1E", "#FFC400"),
    "France":          ("#1D4ED8", "#FFFFFF"),
    "Australia":       ("#00843D", "#FFD700"),
    "Argentina":       ("#74ACDF", "#FFFFFF"),
    "Iceland":         ("#1E40AF", "#FFFFFF"),
    "Peru":            ("#D91023", "#FFFFFF"),
    "Denmark":         ("#C60C30", "#FFFFFF"),
    "Croatia":         ("#FF0000", "#FFFFFF"),
    "Nigeria":         ("#008751", "#FFFFFF"),
    "Costa Rica":      ("#1D4ED8", "#CE1126"),
    "Serbia":          ("#C6363C", "#0C4076"),
    "Germany":         ("#2563EB", "#FACC15"),
    "Mexico":          ("#006847", "#CE1126"),
    "Brazil":          ("#009C3B", "#FFDF00"),
    "Switzerland":     ("#D52B1E", "#FFFFFF"),
    "Sweden":          ("#006AA7", "#FECC02"),
    "South Korea":     ("#C60C30", "#003478"),
    "Belgium":         ("#E30613", "#FAE042"),
    "Panama":          ("#005293", "#DB0000"),
    "Tunisia":         ("#E70013", "#FFFFFF"),
    "England":         ("#FFFFFF", "#CF081F"),
    "Colombia":        ("#FCD116", "#003087"),
    "Japan":           ("#1D4ED8", "#BC002D"),
    "Poland":          ("#FFFFFF", "#DC143C"),
    "Senegal":         ("#00853F", "#FFFFFF"),
    "Netherlands":     ("#F36C21", "#FFFFFF"),
    "Italy":           ("#0066BC", "#FFFFFF"),
    # CLB
    "Real Madrid":     ("#FFFFFF", "#1E3F66"),
    "Barcelona":       ("#A50044", "#004D98"),
    "Manchester City": ("#6CABDD", "#1C2C5B"),
    "Liverpool":       ("#C8102E", "#FFFFFF"),
    "Arsenal":         ("#EF0107", "#FFFFFF"),
    "Chelsea":         ("#034694", "#FFFFFF"),
    "Manchester United":("#DA291C", "#FBE122"),
    "Bayern Munich":   ("#DC052D", "#0066B2"),
    "Bayern München":  ("#DC052D", "#FFFFFF"),
    "Paris Saint-Germain": ("#1D4ED8", "#DA291C"),
    "Tottenham Hotspur": ("#2563EB", "#FFFFFF"),
    "Internazionale":  ("#010E80", "#38BDF8"),
    "AC Milan":        ("#FB090B", "#111827"),
    "Juventus":        ("#111827", "#FFFFFF"),
    "Borussia Dortmund": ("#FDE100", "#111827"),
    "Atlético de Madrid": ("#CB3524", "#FFFFFF"),
    "Bayer 04 Leverkusen": ("#E32221", "#111827"),
    "SL Benfica":      ("#E83030", "#FFFFFF"),
    "FC Porto":        ("#00428C", "#FFFFFF"),
    "SSC Napoli":      ("#12A0D7", "#FFFFFF"),
    "Ajax":            ("#D2122E", "#FFFFFF"),
    "Leipzig":         ("#DD0741", "#FFFFFF"),
    "Atalanta":        ("#1E71B8", "#111827"),
    "Sporting":        ("#008057", "#FFFFFF"),
    "Aston Villa":     ("#670E36", "#95BFE5"),
    "Eibar":           ("#004D98", "#A50044"),
}

def _get_team_colors(team_name):
    if not team_name:
        return ("#1a6edb", "#FFFFFF")
    for k, v in _TEAM_COLORS.items():
        if k.lower() in team_name.lower() or team_name.lower() in k.lower():
            return v
    h = int(hashlib.md5(team_name.encode()).hexdigest()[:6], 16)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return (f"#{r:02x}{g:02x}{b:02x}", "#FFFFFF")

# Tên rút gọn theo cách gọi quen thuộc trên bảng tỷ số và đội hình.
_PLAYER_DISPLAY_ALIASES = {
    "lionel andres messi cuccittini": "Messi",
    "angel fabian di maria hernandez": "Di María",
    "luis jan piers advincula castrillon": "Luis Advíncula",
    "kylian mbappe lottin": "Mbappé",
    "cristiano ronaldo dos santos aveiro": "Cristiano Ronaldo",
    "neymar da silva santos junior": "Neymar",
    "sergio leonel aguero del castillo": "Agüero",
    "gonzalo gerardo higuain": "Higuaín",
    "paulo bruno exequiel dybala": "Dybala",
    "rodrigo javier de paul": "De Paul",
    "damian emiliano martinez": "Dibu Martínez",
    "nahuel molina lucero": "Molina",
    "alexis mac allister": "Mac Allister",
    "enzo fernandez": "Enzo Fernández",
    "enzo jeremias fernandez": "Enzo Fernández",
    "julian alvarez": "J. Álvarez",
    "cristian gabriel romero": "Romero",
    "nicolas hernan otamendi": "Otamendi",
    "nicolas hernan gonzalo otamendi": "Otamendi",
    "nicolas alejandro tagliafico": "Tagliafico",
    "leandro daniel paredes": "Paredes",
    "marcos javier acuna": "Acuña",
    "gonzalo ariel montiel": "Montiel",
    "german alejandro pezzella": "Pezzella",
    "lautaro javier martinez": "Lautaro Martínez",
    "exequiel alejandro palacios": "Palacios",
    "angel martin correa": "Ángel Correa",
    "thiago ezequiel almada": "Thiago Almada",
    "alejandro dario gomez": "Papu Gómez",
    "juan marcos foyth": "Foyth",
    "guido rodriguez": "Guido Rodríguez",
    "geronimo rulli": "Rulli",
    "franco armani": "Armani",
    "lisandro martinez": "Lisandro Martínez",
    "theo bernard francois hernandez": "Theo Hernández",
    "aurelien djani tchouameni": "Tchouaméni",
    "dayotchanculle upamecano": "Upamecano",
    "kingsley coman": "Coman",
    "randal kolo muani": "Kolo Muani",
    "eduardo camavinga": "Camavinga",
    "marcus thuram": "Thuram",
    "youssouf fofana": "Y. Fofana",
    "ibrahima konate": "Konaté",
    "matteo guendouzi olie": "Guendouzi",
    "mohamed salah hamed ghaly": "Salah",
    "mohamed salah": "Salah",
    "roberto firmino barbosa de oliveira": "Firmino",
    "alisson ramses becker": "Alisson",
    "virgil van dijk": "Van Dijk",
    "fabio henrique tavares": "Fabinho",
    "jordan brian henderson": "Henderson",
    "joel andre job matip": "Matip",
    "trent alexander-arnold": "Alexander-Arnold",
    "bamidele alli": "Dele Alli",
    "christian dannemann eriksen": "Eriksen",
    "heung-min son": "Son",
    "gabriel fernando de jesus": "Gabriel Jesus",
    "philippe coutinho correia": "Coutinho",
    "willian borges da silva": "Willian",
    "luis alberto suarez diaz": "Luis Suárez",
    "edinson roberto cavani gomez": "Cavani",
    "james david rodriguez rubio": "James Rodríguez",
    "juan guillermo cuadrado bello": "Cuadrado",
    "radamel falcao garcia zarate": "Falcao",
    "alexis alejandro sanchez sanchez": "Alexis Sánchez",
    "arturo erasmus vidal pardo": "Arturo Vidal",
    "arturo erasmo vidal pardo": "Arturo Vidal",
    "andres iniesta lujan": "Iniesta",
    "xavier hernandez creus": "Xavi",
    "gerard pique bernabeu": "Piqué",
    "marc-andre ter stegen": "Ter Stegen",
    "sergio busquets i burgos": "Busquets",
    "hector junior firpo adames": "Junior Firpo",
    "arthur henrique ramos de oliveira melo": "Arthur Melo",
    "nelson cabral semedo": "Semedo",
    "anssumane fati": "Ansu Fati",
    "sergi roberto carnicer": "Sergi Roberto",
    "frenkie de jong": "De Jong",
    "alvaro tejero sacristan": "Tejero",
    "esteban rodrigo burgos": "E. Burgos",
    "anaitz arbilla zabala": "Arbilla",
    "jose angel valdes diaz": "José Ángel",
    "papa kouly diop": "Diop",
    "eduardo exposito jaen": "Expósito",
    "fabian ariel orellana valenzuela": "Orellana",
    "sergio enrich ametller": "Sergi Enrich",
    "enrique gonzalez casin": "Quique González",
    "pedro leon sanchez gil": "Pedro León",
    "pablo ezequiel de blasis": "De Blasis",
    "paulo andre rodrigues de oliveira": "Paulo Oliveira",
    "sebastian carlos cristoforo pepe": "Cristóforo",
    "sergio alvarez diaz": "Sergio Álvarez",
    "yoel rodriguez oterino": "Yoel",
    "unai simon mendibil": "Unai Simón",
    "daniel carvajal ramos": "Carvajal",
    "robin aime robert le normand": "Le Normand",
    "marc cucurella saseta": "Cucurella",
    "rodrigo hernandez cascante": "Rodri",
    "fabian ruiz pena": "Fabián Ruiz",
    "lamine yamal nasraoui ebana": "Lamine Yamal",
    "daniel olmo carvajal": "Dani Olmo",
    "nicholas williams arthuer": "Nico Williams",
    "alvaro borja morata martin": "Morata",
    "mikel oyarzabal ugarte": "Oyarzabal",
    "martin zubimendi ibanez": "Zubimendi",
    "mikel merino zazon": "Mikel Merino",
    "jose ignacio fernandez iglesias": "Nacho",
    "ferran torres garcia": "Ferran Torres",
    "alejandro grimaldo garcia": "Grimaldo",
    "alejandro baena rodriguez": "Álex Baena",
    "fermin lopez marin": "Fermín López",
    "jesus navas gonzalez": "Jesús Navas",
    "jose luis sanmartin mato": "Joselu",
    "daniel vivian moreno": "Vivian",
    "david raya martin": "David Raya",
    "alejandro remiro gargallo": "Álex Remiro",
    "fernando jose torres sanz": "Fernando Torres",
    "sergio ramos garcia": "Sergio Ramos",
    "raheem shaquille sterling": "Sterling",
    "paul labile pogba": "Pogba",
    "n'golo kante": "Kanté",
    "lucas hernandez pi": "Lucas Hernández",
    "samuel yves umtiti": "Umtiti",
    "cristian david pavon": "Pavón",
    "luka modric": "Modrić",
    "zlatan ibrahimovic": "Ibrahimović",
    "steven n'kemboanza mike christopher nzonzi": "Nzonzi",
    # Brazil, Spain, Portugal, Belgium, Uruguay, Colombia, Argentina (WC 2018)
    "carlos henrique casimiro": "Casemiro",
    "fernando luiz rosa": "Fernandinho",
    "jose paulo bezzera maciel junior": "Paulinho",
    "jose paulo bezerra maciel junior": "Paulinho",
    "marcos aoas correa": "Marquinhos",
    "marcelo vieira da silva junior": "Marcelo",
    "joao miranda de souza filho": "Miranda",
    "danilo luiz da silva": "Danilo",
    "douglas costa de souza": "Douglas Costa",
    "filipe luis kasmirski": "Filipe Luís",
    "fagner conserva lemos": "Fágner",
    "renato soares de oliveira augusto": "Renato Augusto",
    "thiago emiliano da silva": "Thiago Silva",
    "francisco roman alarcon suarez": "Isco",
    "jorge resurreccion merodio": "Koke",
    "david de gea quintana": "De Gea",
    "jordi alba ramos": "Jordi Alba",
    "marco asensio willemsen": "Marco Asensio",
    "thiago alcantara do nascimento": "Thiago Alcântara",
    "iago aspas juncal": "Iago Aspas",
    "lucas vazquez iglesias": "Lucas Vázquez",
    "rodrigo moreno machado": "Rodrigo",
    "david josue jimenez silva": "David Silva",
    "diego da silva costa": "Diego Costa",
    "kleper laveran lima ferreira": "Pepe",
    "kepler laveran lima ferreira": "Pepe",
    "ricardo andrade quaresma bernardo": "Quaresma",
    "joao mario naval da costa eduardo": "João Mário",
    "bernardo mota veiga de carvalho e silva": "Bernardo Silva",
    "bruno miguel borges fernandes": "Bruno Fernandes",
    "rui pedro dos santos patricio": "Rui Patrício",
    "jose miguel da rocha fonte": "José Fonte",
    "william silva de carvalho": "William Carvalho",
    "raphael adelino jose guerreiro": "Guerreiro",
    "goncalo manuel ganchinho guedes": "Gonçalo Guedes",
    "joao filipe iria santos moutinho": "João Moutinho",
    "cedric ricardo alves soares": "Cédric",
    "ricardo domingos barbosa pereira": "Ricardo Pereira",
    "andre miguel valente silva": "André Silva",
    "gelson dany batalha martins": "Gelson Martins",
    "adrien sebastian perruchet silva": "Adrien Silva",
    "manuel henrique tavares fernandes": "Manuel Fernandes",
    "kevin de bruyne": "De Bruyne",
    "romelu lukaku menama": "Lukaku",
    "michy batshuayi tunga": "Batshuayi",
    "yannick ferreira carrasco": "Carrasco",
    "moussa sidi yaya dembele": "Mousa Dembélé",
    "anga dedryck boyata": "Boyata",
    "daniel nii tackie mensah welbeck": "Welbeck",
    "mario gomez garcia": "Mario Gómez",
    "faustino marcos alberto rojo": "Marcos Rojo",
    "javier alejandro mascherano": "Mascherano",
    "ever maximiliano david banega": "Banega",
    "wilfredo daniel caballero": "Caballero",
    "enzo nicolas perez": "Enzo Pérez",
    "lucas rodrigo biglia": "Biglia",
    "gabriel ivan mercado": "Mercado",
    "federico julian fazio": "Fazio",
    "eduardo antonio salvio": "Salvio",
    "maximiliano eduardo meza": "Meza",
    "diego roberto godin leal": "Godín",
    "jose maria gimenez de vargas": "Giménez",
    "nestor fernando muslera micol": "Muslera",
    "rodrigo bentancur colman": "Bentancur",
    "lucas torreira di pascua": "Torreira",
    "giorgian daniel de arrascaeta benedetti": "De Arrascaeta",
    "jose martin caceres silva": "Cáceres",
    "cristhian ricardo stuani curbelo": "Stuani",
    "matias vecino falero": "Vecino",
    "nahitan michel nandez acosta": "Nández",
    "sebastian coates nion": "Coates",
    "diego sebastian laxalt suarez": "Laxalt",
    "maximiliano gomez gonzalez": "Maxi Gómez",
    "cristian gabriel rodriguez barrotti": "C. Rodríguez",
    "david ospina ramirez": "Ospina",
    "yerry fernando mina gonzalez": "Yerry Mina",
    "davinson sanchez mina": "Davinson Sánchez",
    "juan fernando quintero paniagua": "Quintero",
    "carlos arturo bacca ahumada": "Bacca",
    "luis fernando muriel fruto": "Muriel",
    "jefferson andres lerma solis": "Lerma",
    "wilmar enrique barrios teran": "Barrios",
    "santiago arias naranjo": "Arias",
    "johan andres mojica palacio": "Mojica",
    "andres mateus uribe villa": "Mateus Uribe",
    "jose paolo guerrero gonzalez": "Paolo Guerrero",
    "mathias jattah-njie jorgensen": "Zanka",
}


def _compact_html(raw_html):
    """Nén HTML thành 1 dòng (bỏ thụt lề & dòng trống) để Markdown không biến thành khối code."""
    return "".join(line.strip() for line in str(raw_html).splitlines() if line.strip())


def _normalize_player_name(name):
    text = str(name or "").replace("’", "'").replace("‘", "'").replace('"', "'")
    ascii_name = _ud.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.casefold().split())


def _player_display_name(name, pitch=False):
    full_name = " ".join(str(name or "").split())
    if not full_name:
        return "Cầu thủ"

    alias = _PLAYER_DISPLAY_ALIASES.get(_normalize_player_name(full_name))
    if alias:
        return alias

    parts = full_name.split()
    if pitch and len(parts) > 1:
        return parts[-1]
    if len(parts) > 2:
        return f"{parts[0]} {parts[-1]}"
    return full_name


def _make_jersey_svg(initials, primary_color, secondary_color, size=80):
    s = size
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 80 80">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{primary_color};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{primary_color}cc;stop-opacity:1"/>
    </linearGradient>
    <clipPath id="circ"><circle cx="40" cy="40" r="38"/></clipPath>
  </defs>
  <circle cx="40" cy="40" r="39" fill="{primary_color}" opacity="0.15"/>
  <circle cx="40" cy="40" r="38" fill="url(#bg)"/>
  <g clip-path="url(#circ)">
    <path d="M20,30 L14,20 L28,16 Q40,22 52,16 L66,20 L60,30 L54,28 L54,64 L26,64 L26,28 Z"
          fill="{secondary_color}" opacity="0.9"/>
    <path d="M28,16 Q34,26 40,24 Q46,26 52,16" fill="{primary_color}" opacity="0.3"/>
  </g>
  <circle cx="40" cy="40" r="38" fill="none" stroke="{secondary_color}" stroke-width="1.5" opacity="0.5"/>
  <text x="40" y="50" font-size="20" font-weight="900" text-anchor="middle"
        fill="{primary_color}" font-family="Arial Black,Arial,sans-serif"
        stroke="{primary_color}" stroke-width="0.5">{initials}</text>
  <text x="40" y="49" font-size="20" font-weight="900" text-anchor="middle"
        fill="{secondary_color}" font-family="Arial Black,Arial,sans-serif">{initials}</text>
</svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

_PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_photos")
_PHOTO_INDEX_PATH = os.path.join(_PHOTO_DIR, "photo_index.json")

def _norm_name(s):
    s = "".join(c for c in _ud.normalize("NFKD", str(s)) if not _ud.combining(c))
    return " ".join(s.lower().replace('"', "'").split())

def _photo_name_key(name):
    """Create a stable accent- and punctuation-insensitive key for photo lookup."""
    return "".join(ch for ch in _norm_name(name) if ch.isalnum())

def _load_photo_indices():
    idx = {}
    if os.path.exists(_PHOTO_INDEX_PATH):
        try:
            with open(_PHOTO_INDEX_PATH, "r", encoding="utf-8") as _f:
                idx = json.load(_f)
        except Exception:
            idx = {}
    idx_norm = {_norm_name(k): v for k, v in idx.items()}
    idx_key = {}
    for _photo_name, _photo_file in idx.items():
        idx_key.setdefault(_photo_name_key(_photo_name), _photo_file)
    file_idx = {}
    if os.path.isdir(_PHOTO_DIR):
        for _photo_file in os.listdir(_PHOTO_DIR):
            if os.path.splitext(_photo_file)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                _photo_key = _photo_name_key(os.path.splitext(_photo_file)[0])
                if _photo_key:
                    file_idx.setdefault(_photo_key, _photo_file)
    return idx, idx_norm, idx_key, file_idx

_PHOTO_INDEX, _PHOTO_INDEX_NORM, _PHOTO_INDEX_KEY, _PHOTO_FILE_INDEX = _load_photo_indices()

@st.cache_data(show_spinner=False)
def _load_photo_b64(fpath):
    try:
        with open(fpath, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(fpath)[1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None

def get_player_avatar(p_name, is_home, team_name=""):
    global _PHOTO_INDEX, _PHOTO_INDEX_NORM, _PHOTO_INDEX_KEY, _PHOTO_FILE_INDEX
    if not p_name:
        p_name = "Unknown"
    photo_key = _photo_name_key(p_name)
    fname = (
        _PHOTO_INDEX.get(p_name)
        or _PHOTO_INDEX_NORM.get(_norm_name(p_name))
        or _PHOTO_INDEX_KEY.get(photo_key)
        or _PHOTO_FILE_INDEX.get(photo_key)
    )
    if not fname:
        _PHOTO_INDEX, _PHOTO_INDEX_NORM, _PHOTO_INDEX_KEY, _PHOTO_FILE_INDEX = _load_photo_indices()
        fname = (
            _PHOTO_INDEX.get(p_name)
            or _PHOTO_INDEX_NORM.get(_norm_name(p_name))
            or _PHOTO_INDEX_KEY.get(photo_key)
            or _PHOTO_FILE_INDEX.get(photo_key)
        )
    if fname:
        fpath = fname if os.path.isabs(fname) else os.path.join(_PHOTO_DIR, os.path.basename(fname))
        if os.path.exists(fpath):
            b64 = _load_photo_b64(fpath)
            if b64:
                return b64

    parts = p_name.split()
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        initials = parts[0][:2].upper()
    else:
        initials = "FB"
    
    pri, sec = _get_team_colors(team_name) if team_name else (
        ("#1a6edb", "#FFFFFF") if is_home else ("#cc2255", "#FFFFFF")
    )
    return _make_jersey_svg(initials, pri, sec)


# ==========================================================
# 🧠 CÁC HÀM TÍNH TOÁN & MÔ HÌNH HỌC TOÁN HỌC (ANALYTICS ENGINE)
# ==========================================================

def estimate_xg(x, y):
    if pd.isna(x) or pd.isna(y):
        return 0.1
    dx = 120.0 - float(x)
    dy = 40.0 - float(y)
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(8.0 * dx, dx**2 + dy**2 - 16.0)
    angle = np.maximum(angle, 0.05)
    logit = 0.85 - 0.095 * dist + 1.25 * angle
    prob = 1.0 / (1.0 + np.exp(-logit))
    return round(float(np.clip(prob, 0.01, 0.95)), 2)

def _sec(ts):
    parts = str(ts).split(':')
    if len(parts) == 3:
        try:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except Exception:
            return 0.0
    return 0.0

def _true_mask(values):
    return values.astype("string").str.strip().str.lower().isin({"1", "1.0", "true", "yes", "y"})

def _pass_success_mask(pass_events):
    if 'Successful Passes' in pass_events.columns:
        return pd.to_numeric(pass_events['Successful Passes'], errors="coerce").fillna(0).gt(0)
    if 'pass_outcome_name' in pass_events.columns:
        outcome = pass_events['pass_outcome_name'].astype("string").str.strip().str.lower()
        return outcome.isna() | outcome.isin({"", "complete", "completed", "successful"})
    return None

def _sum_numeric(frame, column):
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.sum()) if values.notna().any() else None

def _count_flag(frame, column):
    if column not in frame.columns:
        return None
    return int(_true_mask(frame[column]).sum())

def _count_indicator(frame, column):
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return int(values.fillna(0).gt(0).sum())

def _estimate_zone_xt(x_series, y_series):
    x = pd.to_numeric(x_series, errors="coerce").clip(0, 120)
    y = pd.to_numeric(y_series, errors="coerce").clip(0, 80)
    return 0.0038 * np.exp(0.034 * x) * np.exp(-0.0012 * ((y - 40.0) ** 2))

def _pass_summary(pass_events):
    attempted = len(pass_events)
    success_mask = _pass_success_mask(pass_events)
    completed = int(success_mask.sum()) if success_mask is not None else None
    accuracy = round(completed / attempted * 100, 1) if attempted and completed is not None else None
    pressure_mask = _true_mask(pass_events['under_pressure']) if 'under_pressure' in pass_events.columns else None
    pressure_attempts = int(pressure_mask.sum()) if pressure_mask is not None else None
    pressure_completed = int(success_mask[pressure_mask].sum()) if pressure_mask is not None and success_mask is not None else None

    progressive = _count_indicator(pass_events, 'Progressive Passes')
    progressive_completed = _count_indicator(pass_events, 'Successful Progressive Passes')
    if progressive is None and {'location_x', 'pass_end_location_x'}.issubset(pass_events.columns):
        dx = pd.to_numeric(pass_events['pass_end_location_x'], errors="coerce") - pd.to_numeric(pass_events['location_x'], errors="coerce")
        prog_mask = dx.fillna(0) >= 12.0
        progressive = int(prog_mask.sum())
        if success_mask is not None:
            progressive_completed = int((prog_mask & success_mask).sum())
        else:
            progressive_completed = progressive

    key_passes = _count_flag(pass_events, 'pass_shot_assist')
    if key_passes is None and 'pass_assisted_shot_id' in pass_events.columns:
        key_passes = int(pass_events['pass_assisted_shot_id'].notna().sum())

    crosses = _count_indicator(pass_events, 'Crosses') if 'Crosses' in pass_events.columns else _count_flag(pass_events, 'pass_cross')
    crosses_completed = _count_indicator(pass_events, 'Successful Crosses')
    if crosses_completed is None and 'pass_cross' in pass_events.columns and success_mask is not None:
        crosses_completed = int((_true_mask(pass_events['pass_cross']) & success_mask).sum())

    throughballs = _count_indicator(pass_events, 'Throughballs') if 'Throughballs' in pass_events.columns else _count_flag(pass_events, 'pass_through_ball')
    throughballs_completed = _count_indicator(pass_events, 'Successful Throughballs')
    if throughballs_completed is None and 'pass_through_ball' in pass_events.columns and success_mask is not None:
        throughballs_completed = int((_true_mask(pass_events['pass_through_ball']) & success_mask).sum())

    xt_val = _sum_numeric(pass_events, 'xT')
    if xt_val is None and {'location_x', 'location_y', 'pass_end_location_x', 'pass_end_location_y'}.issubset(pass_events.columns):
        start_xt = _estimate_zone_xt(pass_events['location_x'], pass_events['location_y'])
        end_xt = _estimate_zone_xt(pass_events['pass_end_location_x'], pass_events['pass_end_location_y'])
        d_xt = (end_xt - start_xt).where(success_mask if success_mask is not None else True, 0.0)
        if d_xt.notna().any():
            xt_val = float(d_xt.fillna(0.0).sum())

    return {
        "attempted": attempted,
        "completed": completed,
        "accuracy": accuracy,
        "progressive": progressive,
        "progressive_completed": progressive_completed,
        "key_passes": key_passes,
        "assists": _count_flag(pass_events, 'pass_goal_assist'),
        "crosses": crosses,
        "crosses_completed": crosses_completed,
        "throughballs": throughballs,
        "throughballs_completed": throughballs_completed,
        "switches": _count_indicator(pass_events, 'Switches') if 'Switches' in pass_events.columns else _count_flag(pass_events, 'pass_switch'),
        "xt": xt_val,
        "xt_facilitated": _sum_numeric(pass_events, 'xT Facilitated'),
        "avg_length": float(pd.to_numeric(pass_events['pass_length'], errors="coerce").mean()) if 'pass_length' in pass_events.columns and pd.to_numeric(pass_events['pass_length'], errors="coerce").notna().any() else None,
        "pressure_attempts": pressure_attempts,
        "pressure_completed": pressure_completed,
    }

def _display_count(value):
    return "Chưa có" if value is None else f"{value:,}"

def _display_decimal(value, digits=2, signed=False):
    if value is None or pd.isna(value):
        return "Chưa có"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

def _format_stat_val(data, column, digits=0, suffix=""):
    if data is None:
        return "Chưa có dữ liệu"
    val = None
    if isinstance(data, dict):
        val = data.get(column)
    elif hasattr(data, 'index') and column in data.index:
        val = data[column]
    if val is None or pd.isna(val) or val == "":
        return "Chưa có dữ liệu"
    try:
        num = float(val)
        if digits == 0:
            return f"{int(round(num)):,}{suffix}"
        return f"{num:.{digits}f}{suffix}"
    except Exception:
        return str(val) + suffix

def calculate_player_rating(p_events, is_gk=False, gk_saves_override=0, goals_conceded=0):
    if p_events.empty:
        return 6.0, {}, 0, 0.0, 0, 0, 0

    base = 6.3 if is_gk else 6.0
    p_shots = p_events[p_events['type_name'] == 'Shot'] if 'type_name' in p_events.columns else pd.DataFrame()
    goals = len(p_shots[p_shots['shot_outcome_name'] == 'Goal']) if 'shot_outcome_name' in p_shots.columns else 0
    shots_on_target = len(p_shots[p_shots.get('shot_outcome_name', '').isin(['Goal', 'Saved', 'Saved to Post'])]) if not p_shots.empty and 'shot_outcome_name' in p_shots.columns else 0

    xg_vals = pd.to_numeric(p_events.get('shot_statsbomb_xg'), errors='coerce').dropna() if 'shot_statsbomb_xg' in p_events.columns else pd.Series(dtype=float)
    xg = float(xg_vals.sum()) if not xg_vals.empty else 0.0

    p_passes = p_events[p_events['type_name'] == 'Pass'] if 'type_name' in p_events.columns else pd.DataFrame()
    tot_passes = len(p_passes)
    if 'pass_outcome_name' in p_passes.columns and p_passes['pass_outcome_name'].notna().any():
        comp_passes = len(p_passes[p_passes['pass_outcome_name'].isin(['', 'Complete', 'completed']) | p_passes['pass_outcome_name'].isna()])
    elif 'Successful Passes' in p_passes.columns:
        comp_passes = int(pd.to_numeric(p_passes['Successful Passes'], errors='coerce').fillna(0).sum())
    elif 'pass_outcome_name' in p_passes.columns:
        comp_passes = len(p_passes[p_passes['pass_outcome_name'].isin(['', 'Complete', 'completed']) | p_passes['pass_outcome_name'].isna()])
    else:
        comp_passes = tot_passes

    pass_pct = (comp_passes / tot_passes * 100) if tot_passes > 0 else 75.0
    key_passes = len(p_passes[pd.to_numeric(p_passes.get('pass_shot_assist'), errors='coerce').fillna(0).eq(1)]) if 'pass_shot_assist' in p_passes.columns else 0
    if key_passes == 0 and 'pass_assisted_shot_id' in p_passes.columns:
        key_passes = int(p_passes['pass_assisted_shot_id'].notna().sum())
    assists = len(p_passes[pd.to_numeric(p_passes.get('pass_goal_assist'), errors='coerce').fillna(0).eq(1)]) if 'pass_goal_assist' in p_passes.columns else 0

    defense_types = ['Duel', 'Interception', 'Ball Recovery', 'Block', 'Clearance']
    tackles = len(p_events[p_events.get('type_name', '').isin(defense_types)]) if 'type_name' in p_events.columns else 0
    if tackles == 0 and 'previous_event' in p_events.columns:
        tackles = int(p_events['previous_event'].fillna('').astype(str).isin(defense_types).sum())
    fouls = len(p_events[p_events.get('type_name', '') == 'Foul Committed']) if 'type_name' in p_events.columns else 0

    # Bonus từ chuyền tịnh tiến / kéo bóng / xT (hỗ trợ cả World Cup 2018 & StatsBomb)
    prog_passes = int(pd.to_numeric(p_passes['Successful Progressive Passes'], errors='coerce').fillna(0).sum()) if 'Successful Progressive Passes' in p_passes.columns else 0
    if prog_passes == 0 and {'location_x', 'pass_end_location_x'}.issubset(p_passes.columns):
        dx = pd.to_numeric(p_passes['pass_end_location_x'], errors='coerce') - pd.to_numeric(p_passes['location_x'], errors='coerce')
        prog_passes = int((dx.fillna(0) >= 15.0).sum())
    prog_carries = int(pd.to_numeric(p_events['Progressive Carries'], errors='coerce').fillna(0).sum()) if 'Progressive Carries' in p_events.columns else 0
    xt_sum = float(pd.to_numeric(p_passes['xT'], errors='coerce').fillna(0).sum()) if 'xT' in p_passes.columns else 0.0

    b_gk = 0.0
    if is_gk:
        gk_saves = 0
        gk_claims = 0
        if 'gk_type_name' in p_events.columns:
            gk_t = p_events['gk_type_name'].fillna('').astype(str)
            gk_saves = int(gk_t.str.contains('Saved', case=False, regex=False).sum())
            gk_claims = int(gk_t.isin(['Collected', 'Punch', 'Keeper Sweeper', 'Smother']).sum())
        if gk_saves == 0 and gk_saves_override:
            gk_saves = int(gk_saves_override)
        if gk_claims == 0 and 'previous_event' in p_events.columns:
            gk_claims = int((p_events['previous_event'] == 'Goal Keeper').sum())
        cs_bonus = 0.35 if goals_conceded == 0 else max(-0.45, -int(goals_conceded) * 0.12)
        b_gk = min(2.4, round(gk_saves * 0.26 + gk_claims * 0.06 + cs_bonus, 2))

    b_goals = min(3.0, goals * 1.0)
    b_assists = min(1.6, assists * 0.8)
    b_xg = min(1.2, round(xg * 0.4, 2))
    b_shots = min(0.8, shots_on_target * 0.15)
    b_key_passes = min(1.0, key_passes * 0.25)
    b_passes = 0.35 if pass_pct >= 85 and tot_passes >= 15 else (0.15 if pass_pct >= 75 and tot_passes >= 10 else (-0.25 if pass_pct < 65 and tot_passes >= 15 else 0.05))
    b_progression = min(0.65, round(prog_passes * 0.04 + prog_carries * 0.05 + max(0.0, xt_sum) * 1.2, 2))
    b_tackles = min(1.2, tackles * 0.08) + b_progression
    p_fouls = min(0.6, fouls * 0.08)

    final_score = base + b_gk + b_goals + b_assists + b_xg + b_shots + b_key_passes + b_passes + b_tackles - p_fouls
    final_score = max(5.0, min(10.0, round(final_score, 1)))

    breakdown = {
        "Base (Khởi điểm)": base, "Bàn thắng": b_goals, "Kiến tạo": b_assists,
        "xG (Kỳ vọng)": b_xg, "Sút trúng đích": b_shots,
        "Key pass (Tạo cơ hội)": b_key_passes, "Chuyền bóng": b_passes,
        "Phòng ngự & Tịnh tiến": round(b_tackles + b_gk, 2), "Phạm lỗi": -p_fouls
    }
    return final_score, breakdown, goals, xg, tot_passes, comp_passes, tackles


def _make_impact_bar(label, impact, pos_color="#00F0FF"):
    pct = int(abs(impact) * 48)
    pct = max(4, pct) if abs(impact) > 0.05 else 0
    if impact >= 0.05:
        bar_html = f'<div style="position: absolute; left: 50%; width: {pct}%; height: 100%; background: {pos_color}; border-radius: 0 3px 3px 0;"></div>'
    elif impact <= -0.05:
        bar_html = f'<div style="position: absolute; right: 50%; width: {pct}%; height: 100%; background: #F97316; border-radius: 3px 0 0 3px;"></div>'
    else:
        bar_html = ''
    value_color = pos_color if impact >= 0.05 else ('#F97316' if impact <= -0.05 else '#94A3B8')
    impact_value = f'{impact:+.2f}'
    return (
        '<div style="display: grid; grid-template-columns: 78px minmax(0, 1fr) 42px; align-items: center; gap: 7px; padding: 2px 0;">'
        f'<span style="font-size: 12px; color: #CBD5E1; font-weight: 600; white-space: nowrap;">{label}</span>'
        '<div style="height: 6px; background: #1C2536; border-radius: 4px; position: relative;">'
        '<div style="position: absolute; left: 50%; top: -2px; bottom: -2px; width: 2px; background: #94A3B8; z-index: 2;"></div>'
        f'{bar_html}'
        '</div>'
        f'<span style="text-align: right; color: {value_color}; font-size: 11.5px; font-weight: 800; font-variant-numeric: tabular-nums;">{impact_value}</span>'
        '</div>'
    )


def _render_stat_row(name, value, note=None):
    note_html = f' <span style="color:#64748B; font-size:11px;">({note})</span>' if note else ''
    return f'<div class="sofa-stat-row"><span class="stat-name">{html.escape(str(name))}{note_html}</span><span class="stat-val">{html.escape(str(value))}</span></div>'


def render_sofascore_player_panel(player_data, p_events, current_match, max_rating=None, match_events=None):
    if not player_data:
        st.info("💡 Chọn một cầu thủ để xem phân tích chi tiết.")
        return

    p_name = player_data.get('name', '')
    disp_name = player_data.get('display_name', p_name)
    pos = player_data.get('pos', 'Cầu thủ')
    is_gk = "goalkeeper" in str(pos).lower() or str(pos) == "GK"
    rating = player_data.get('rating', 6.0)
    photo = player_data.get('photo', '')
    team_name = player_data.get('team_name', current_match.get('home_team', ''))
    is_home = (team_name == current_match.get('home_team', ''))

    # 1. Số phút thi đấu chính xác (tính cả hiệp phụ 120' và thời điểm thay người vào/ra)
    has_et = False
    if match_events is not None and not match_events.empty and 'period' in match_events.columns:
        has_et = bool(pd.to_numeric(match_events['period'], errors='coerce').isin([3, 4]).any())
    full_match_mins = 120 if has_et else 90

    min_in = 0
    min_out = full_match_mins
    subs_out = p_events[p_events['type_name'] == 'Substitution'] if 'type_name' in p_events.columns else pd.DataFrame()
    if not subs_out.empty and 'minute' in subs_out.columns:
        min_out = int(pd.to_numeric(subs_out['minute'], errors='coerce').fillna(full_match_mins).iloc[0])

    if player_data.get('is_sub'):
        sub_in_found = False
        if match_events is not None and not match_events.empty and {'type_name', 'substitution_replacement_name', 'minute'}.issubset(match_events.columns):
            sub_in_rows = match_events[
                (match_events['type_name'] == 'Substitution') &
                (match_events['substitution_replacement_name'].astype(str).str.strip() == p_name)
            ]
            if not sub_in_rows.empty:
                min_in = int(pd.to_numeric(sub_in_rows['minute'], errors='coerce').fillna(60).iloc[0])
                sub_in_found = True
        if not sub_in_found:
            mins_series = pd.to_numeric(p_events['minute'], errors='coerce').dropna() if 'minute' in p_events.columns else pd.Series(dtype=float)
            min_in = max(1, int(mins_series.min()) - 1) if not mins_series.empty else 65
    else:
        if subs_out.empty and (match_events is None or 'substitution_replacement_name' not in match_events.columns):
            mins_series = pd.to_numeric(p_events['minute'], errors='coerce').dropna() if 'minute' in p_events.columns else pd.Series(dtype=float)
            if not mins_series.empty:
                max_ev_min = int(mins_series.max())
                if (not is_gk and max_ev_min <= 85) or (is_gk and max_ev_min <= 45):
                    min_out = max_ev_min + 1

    mins_played = max(1, min_out - min_in)

    # 2. Trích xuất chỉ số từ dữ liệu sự kiện (tương thích cả StatsBomb JSON & World Cup 2018 CSV)
    p_shots = p_events[p_events['type_name'] == 'Shot'] if 'type_name' in p_events.columns else pd.DataFrame()
    tot_shots = len(p_shots)
    if not p_shots.empty and 'shot_outcome_name' in p_shots.columns:
        s_out = p_shots['shot_outcome_name'].fillna('').astype(str).str.strip()
        goals = len(p_shots[s_out == 'Goal'])
        shots_on_target = len(p_shots[s_out.isin(['Goal', 'Saved', 'Saved to Post'])])
        shots_blocked = len(p_shots[s_out == 'Blocked'])
        shots_off_target = len(p_shots[s_out.isin(['Off T', 'Wayward', 'Post', 'Saved Off T'])])
    else:
        goals = int(player_data.get('goals') or player_data.get('goals_count') or 0)
        shots_on_target = 0
        shots_blocked = 0
        shots_off_target = 0

    if goals == 0 and (player_data.get('goals') or player_data.get('goals_count')):
        goals = int(player_data.get('goals') or player_data.get('goals_count') or 0)

    xg_vals = pd.to_numeric(p_shots.get('shot_statsbomb_xg'), errors='coerce').dropna() if not p_shots.empty and 'shot_statsbomb_xg' in p_shots.columns else pd.Series(dtype=float)
    xg = round(float(xg_vals.sum()), 2) if not xg_vals.empty else float(player_data.get('xg') or 0.0)

    p_passes = p_events[p_events['type_name'] == 'Pass'] if 'type_name' in p_events.columns else pd.DataFrame()
    pass_summary = _pass_summary(p_passes)
    tot_passes = pass_summary['attempted']
    comp_passes = pass_summary['completed'] if pass_summary['completed'] is not None else tot_passes
    pass_pct = pass_summary['accuracy'] if pass_summary['accuracy'] is not None else 0.0

    key_passes = pass_summary['key_passes'] or 0
    assists = pass_summary['assists'] or 0
    xa = round(key_passes * 0.18 + assists * 0.35, 2)

    crosses_tot = pass_summary['crosses'] or 0
    crosses_comp = pass_summary['crosses_completed'] if pass_summary['crosses_completed'] is not None else 0

    if 'Long Passes' in p_passes.columns:
        long_tot = _count_indicator(p_passes, 'Long Passes') or 0
        long_comp = _count_indicator(p_passes, 'Successful Long Passes') or 0
    else:
        long_passes = p_passes[pd.to_numeric(p_passes.get('pass_length'), errors='coerce').fillna(0) > 30] if 'pass_length' in p_passes.columns else pd.DataFrame()
        long_tot = len(long_passes)
        long_mask = _pass_success_mask(long_passes) if not long_passes.empty else None
        long_comp = int(long_mask.sum()) if long_mask is not None else long_tot

    if 'Final Third Passes' in p_passes.columns:
        f3_tot = _count_indicator(p_passes, 'Final Third Passes') or 0
        f3_comp = _count_indicator(p_passes, 'Successful Final Third Passes') or 0
    elif 'pass_end_location_x' in p_passes.columns:
        f3_passes = p_passes[pd.to_numeric(p_passes['pass_end_location_x'], errors='coerce').fillna(0) >= 80.0]
        f3_tot = len(f3_passes)
        f3_mask = _pass_success_mask(f3_passes) if not f3_passes.empty else None
        f3_comp = int(f3_mask.sum()) if f3_mask is not None else f3_tot
    else:
        f3_tot, f3_comp = 0, 0

    through_tot = pass_summary['throughballs'] or 0
    prog_passes_tot = pass_summary['progressive'] or 0
    prog_passes_comp = pass_summary['progressive_completed'] or 0
    xt_val = pass_summary['xt'] or 0.0

    touch_types = ['Pass', 'Shot', 'Carry', 'Dribble', 'Ball Receipt*', 'Dispossessed', 'Miscontrol', 'Interception', 'Clearance', 'Block', 'Ball Recovery', 'Goal Keeper']
    p_touches = p_events[p_events['type_name'].isin(touch_types)] if 'type_name' in p_events.columns else p_events
    touches = len(p_touches) if not p_touches.empty else len(p_events)
    if 'isTouch' in p_events.columns:
        touches = max(touches, int(pd.to_numeric(p_events['isTouch'], errors='coerce').fillna(0).sum()))

    miscontrols = len(p_events[p_events.get('type_name', '') == 'Miscontrol']) if 'type_name' in p_events.columns else 0
    dispossessed = len(p_events[p_events.get('type_name', '') == 'Dispossessed']) if 'type_name' in p_events.columns else 0
    if miscontrols == 0 and 'previous_event' in p_events.columns:
        miscontrols = int((p_events['previous_event'] == 'Miscontrol').sum())
    incomp_passes = max(0, tot_passes - comp_passes)
    possession_lost = miscontrols + dispossessed + incomp_passes

    p_dribbles = p_events[p_events['type_name'] == 'Dribble'] if 'type_name' in p_events.columns else pd.DataFrame()
    dribbles_tot = len(p_dribbles)
    if not p_dribbles.empty and 'dribble_outcome_name' in p_dribbles.columns:
        dribbles_succ = len(p_dribbles[p_dribbles['dribble_outcome_name'] == 'Complete'])
    else:
        dribbles_succ = int(round(dribbles_tot * 0.6))

    prog_carries = int(pd.to_numeric(p_events['Progressive Carries'], errors='coerce').fillna(0).sum()) if 'Progressive Carries' in p_events.columns else 0
    if prog_carries == 0 and 'type_name' in p_events.columns:
        c_rows = p_events[p_events['type_name'] == 'Carry']
        if 'carry_distance' in c_rows.columns:
            prog_carries = int((pd.to_numeric(c_rows['carry_distance'], errors='coerce').fillna(0) >= 10.0).sum())
    carries_count = int(pd.to_numeric(p_events['Carries'], errors='coerce').fillna(0).sum()) if 'Carries' in p_events.columns else len(p_events[p_events.get('type_name', '') == 'Carry'])

    fouls_won = len(p_events[p_events.get('type_name', '') == 'Foul Won']) if 'type_name' in p_events.columns else 0
    if fouls_won == 0 and 'previous_event' in p_events.columns:
        fouls_won = int((p_events['previous_event'] == 'Foul Won').sum())
    offsides = len(p_events[p_events.get('type_name', '') == 'Offside']) if 'type_name' in p_events.columns else 0

    if 'Carry Distance' in p_events.columns and pd.to_numeric(p_events['Carry Distance'], errors='coerce').notna().any():
        carry_dist_val = pd.to_numeric(p_events['Carry Distance'], errors='coerce').fillna(0).sum()
    elif 'carry_distance' in p_events.columns and pd.to_numeric(p_events['carry_distance'], errors='coerce').notna().any():
        carry_dist_val = pd.to_numeric(p_events['carry_distance'], errors='coerce').fillna(0).sum()
    else:
        carry_dist_val = carries_count * 6.5
    carrying_dist = round(float(carry_dist_val), 1)

    tackles = len(p_events[p_events.get('type_name', '') == 'Duel']) if 'type_name' in p_events.columns else 0
    interceptions = len(p_events[p_events.get('type_name', '') == 'Interception']) if 'type_name' in p_events.columns else 0
    recoveries = len(p_events[p_events.get('type_name', '') == 'Ball Recovery']) if 'type_name' in p_events.columns else 0
    clearances = len(p_events[p_events.get('type_name', '') == 'Clearance']) if 'type_name' in p_events.columns else 0
    blocks = len(p_events[p_events.get('type_name', '') == 'Block']) if 'type_name' in p_events.columns else 0
    if 'previous_event' in p_events.columns:
        prev_ev = p_events['previous_event'].fillna('').astype(str)
        if tackles == 0: tackles = int((prev_ev == 'Duel').sum())
        if interceptions == 0: interceptions = int((prev_ev == 'Interception').sum())
        if recoveries == 0: recoveries = int((prev_ev == 'Ball Recovery').sum())
        if clearances == 0: clearances = int((prev_ev == 'Clearance').sum())
        if blocks == 0: blocks = int((prev_ev == 'Block').sum())

    fouls_committed = len(p_events[p_events.get('type_name', '') == 'Foul Committed']) if 'type_name' in p_events.columns else 0
    yellow_cards = int(player_data.get('yellow_cards') or 0)
    red_cards = int(player_data.get('red_cards') or 0)
    if yellow_cards == 0 and red_cards == 0:
        for c_col in ('card_name', 'foul_committed_card_name', 'bad_behaviour_card_name'):
            if c_col in p_events.columns:
                c_vals = p_events[c_col].fillna('').astype(str).str.casefold()
                yellow_cards += int(c_vals.str.contains('yellow', regex=False).sum())
                red_cards += int(c_vals.str.contains('red|second yellow', regex=True).sum())

    # Chỉ số riêng cho Thủ môn (Goalkeeping)
    gk_saves = int(player_data.get('gk_saves') or 0)
    gk_claims = int(player_data.get('gk_claims') or 0)
    if 'gk_type_name' in p_events.columns and gk_saves == 0:
        gk_t = p_events['gk_type_name'].fillna('').astype(str)
        gk_saves = int(gk_t.str.contains('Saved', case=False, regex=False).sum())
        gk_claims = int(gk_t.isin(['Collected', 'Punch', 'Keeper Sweeper', 'Smother']).sum())
    goals_conceded = int(current_match.get('away_score', 0) if is_home else current_match.get('home_score', 0))

    # Tác động đến điểm số (Impact breakdown)
    imp_shooting = np.clip((goals * 0.6 + (xg - 0.15) * 0.35 + (shots_on_target - 0.3) * 0.2), -1.0, 1.0)
    imp_passing = np.clip((assists * 0.5 + key_passes * 0.22 + (pass_pct - 75.0) * 0.02 + max(0.0, xt_val) * 0.8), -1.0, 1.0)
    imp_dribbling = np.clip((dribbles_succ * 0.28 + prog_carries * 0.08 + fouls_won * 0.15 - (possession_lost - 10) * 0.025), -1.0, 1.0)
    imp_defending = np.clip(((tackles + interceptions + clearances + recoveries) * 0.08 + prog_passes_comp * 0.03 - fouls_committed * 0.15), -1.0, 1.0)
    imp_gk = np.clip((gk_saves * 0.24 + gk_claims * 0.07 + (0.3 if goals_conceded == 0 else -goals_conceded * 0.12)), -1.0, 1.0)

    r_color = "#00D66C" if rating >= 7.5 else ("#FF9800" if rating >= 6.5 else "#EF4444")
    star_icon = " ★" if (max_rating and rating == max_rating) else ""
    role_badge = "Dự bị vào sân" if player_data.get('is_sub') else pos

    # SofaScore Header Card (Compact Pro Layout)
    match_display = f"{current_match.get('home_team')} {current_match.get('home_score')} - {current_match.get('away_score')} {current_match.get('away_team')}"
    st.markdown(
        f'<div style="background: linear-gradient(145deg, #161F2E 0%, #111824 100%); border-radius: 13px; padding: 12px 15px; border: 1.5px solid #26344B; margin-bottom: 8px; box-shadow: 0 5px 16px rgba(0,0,0,0.32);">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; gap: 12px;">'
        f'<div style="display: flex; align-items: center; gap: 12px; min-width: 0; flex: 1;">'
        f'<img src="{photo}" width="54" height="54" style="border-radius: 50%; object-fit: cover; border: 2.5px solid {r_color}; flex-shrink: 0;">'
        f'<div style="min-width: 0; flex: 1;">'
        f'<div style="font-size: 17px; font-weight: 800; color: #FFFFFF; line-height: 1.2; overflow-wrap: anywhere;">{html.escape(disp_name)}</div>'
        f'<div style="margin-top: 4px; display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: 12px; color: #94A3B8;">'
        f'<span>{html.escape(str(role_badge))}</span><span>•</span><span style="color: #CBD5E1; font-weight: 700;">{html.escape(str(team_name))}</span>'
        f'<span>•</span><span>⏱️ <b>{mins_played}\'</b></span>'
        f'</div></div></div>'
        f'<div style="text-align: right; flex-shrink: 0;">'
        f'<div style="background-color: {r_color}; color: #000; font-weight: 900; font-size: 16px; padding: 4px 10px; border-radius: 8px; display: inline-block; line-height: 1.15;">{rating}{star_icon}</div>'
        f'<div style="color: #64748B; font-size: 10.5px; font-weight: 700; margin-top: 3px; text-transform: uppercase;">Rating</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # SofaScore Rating Breakdown Card (2x2 Grid)
    first_bar = (
        _make_impact_bar("Cứu thua", imp_gk, "#00D66C" if imp_gk >= 0 else "#F97316")
        if is_gk else
        _make_impact_bar("Sút bóng", imp_shooting, "#00D66C" if imp_shooting >= 0 else "#F97316")
    )
    pass_bar = _make_impact_bar("Chuyền bóng", imp_passing, "#00F0FF" if imp_passing >= 0 else "#F97316")
    drib_bar = _make_impact_bar("Kéo bóng", imp_dribbling, "#FBBF24" if imp_dribbling >= 0 else "#F97316")
    def_bar = _make_impact_bar("Phòng ngự", imp_defending, "#A78BFA" if imp_defending >= 0 else "#F97316")

    st.markdown(
        f'<div style="background: #0E131E; border-radius: 11px; padding: 10px 13px; border: 1px solid #1C2536; margin-bottom: 8px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
        f'<span style="font-size: 12px; font-weight: 800; color: #E2E8F0; text-transform: uppercase; letter-spacing: 0.04em;">⚡ Tác động ước tính lên điểm</span>'
        f'<span style="font-size: 10.5px; color: #8E9BAE; font-weight: 700;">− GIẢM · + TĂNG</span>'
        f'</div>'
        f'<div class="sofa-impact-grid">{first_bar}{pass_bar}{drib_bar}{def_bar}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Sub-tabs
    sub_stats, sub_shots, sub_heat, sub_radar = st.tabs(["📊 Thống Kê", "🎯 Cú Sút", "🔥 Heatmap", "🕸️ Radar & FIFA"])

    with sub_stats:
        gk_box_html = ""
        if is_gk:
            gk_box_html = (
                f'<div class="sofa-stat-box" style="grid-column: 1 / -1;">'
                f'<div style="color: #34D399; font-weight: 800; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px;">🧤 Thủ Môn</div>'
                f'<div class="sofa-impact-grid">'
                f'{_render_stat_row("Cứu thua", gk_saves)}'
                f'{_render_stat_row("Bắt & đấm bóng", gk_claims)}'
                f'{_render_stat_row("Bàn thua", goals_conceded)}'
                f'{_render_stat_row("Giữ sạch lưới", "Có" if goals_conceded == 0 else "Không")}'
                f'</div></div>'
            )
        st.markdown(
            f'<div class="sofa-stat-grid">'
            f'{gk_box_html}'
            f'<div class="sofa-stat-box">'
            f'<div style="color: #38BDF8; font-weight: 800; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px;">⚡ Tấn Công</div>'
            f'{_render_stat_row("Bàn thắng", goals)}'
            f'{_render_stat_row("Bàn thắng kỳ vọng (xG)", f"{xg:.2f}")}'
            f'{_render_stat_row("Kiến tạo", assists)}'
            f'{_render_stat_row("Kiến tạo kỳ vọng (xA)", f"{xa:.2f}")}'
            f'{_render_stat_row("Tổng cú sút", tot_shots)}'
            f'{_render_stat_row("Sút trúng đích", shots_on_target)}'
            f'{_render_stat_row("Sút bị chặn", shots_blocked)}'
            f'{_render_stat_row("Việt vị", offsides)}'
            f'</div>'
            f'<div class="sofa-stat-box">'
            f'<div style="color: #00D66C; font-weight: 800; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px;">🔄 Chuyền Bóng</div>'
            f'{_render_stat_row("Chuyền chính xác", f"{comp_passes}/{tot_passes} ({pass_pct}%)")}'
            f'{_render_stat_row("Tạo cơ hội (Key pass)", key_passes)}'
            f'{_render_stat_row("Chuyền tịnh tiến", f"{prog_passes_comp}/{prog_passes_tot}")}'
            f'{_render_stat_row("Chuyền vào 1/3 cuối", f"{f3_comp}/{f3_tot}")}'
            f'{_render_stat_row("Độ đe dọa chuyền (xT)", f"{xt_val:+.3f}")}'
            f'{_render_stat_row("Tạt bóng chính xác", f"{crosses_comp}/{crosses_tot}")}'
            f'{_render_stat_row("Bóng dài chính xác", f"{long_comp}/{long_tot}")}'
            f'{_render_stat_row("Chọc khe", through_tot)}'
            f'</div>'
            f'<div class="sofa-stat-box">'
            f'<div style="color: #FBBF24; font-weight: 800; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px;">⚽ Kiểm Soát & Kéo Bóng</div>'
            f'{_render_stat_row("Chạm bóng", touches)}'
            f'{_render_stat_row("Số lần dẫn bóng", carries_count)}'
            f'{_render_stat_row("Kéo bóng tịnh tiến", prog_carries)}'
            f'{_render_stat_row("Quãng đường dẫn bóng", f"{carrying_dist} m")}'
            f'{_render_stat_row("Rê bóng thành công", f"{dribbles_succ}/{dribbles_tot}")}'
            f'{_render_stat_row("Bị phạm lỗi", fouls_won)}'
            f'{_render_stat_row("Khống chế lỗi", miscontrols)}'
            f'{_render_stat_row("Mất quyền kiểm soát", possession_lost)}'
            f'</div>'
            f'<div class="sofa-stat-box">'
            f'<div style="color: #A78BFA; font-weight: 800; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.4px;">🛡️ Phòng Ngự & Kỷ Luật</div>'
            f'{_render_stat_row("Tắc bóng & tranh chấp", tackles)}'
            f'{_render_stat_row("Cắt bóng", interceptions)}'
            f'{_render_stat_row("Giải vây", clearances)}'
            f'{_render_stat_row("Chắn cú sút", blocks)}'
            f'{_render_stat_row("Thu hồi bóng", recoveries)}'
            f'{_render_stat_row("Phạm lỗi", fouls_committed)}'
            f'{_render_stat_row("Thẻ vàng", yellow_cards)}'
            f'{_render_stat_row("Thẻ đỏ", red_cards)}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with sub_shots:
        if tot_shots > 0:
            pills = []
            for _, r in p_shots.iterrows():
                m = int(pd.to_numeric(r.get('minute', 0), errors='coerce') or 0)
                out = str(r.get('shot_outcome_name', '') or '')
                ic = "⚽" if out == "Goal" else ("🧤" if out in ["Saved", "Saved to Post"] else ("🛡️" if out == "Blocked" else "❌"))
                pills.append(f'<span style="background: #1E293B; border: 1px solid #334155; padding: 3px 9px; border-radius: 14px; font-size: 11.5px; font-weight: 700; color: #FFFFFF;">{m}\' {ic}</span>')
            st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px;">{"".join(pills)}</div>', unsafe_allow_html=True)

            fig_mini = go.Figure()
            fig_mini.add_shape(type="rect", x0=75, y0=10, x1=120, y1=70, fillcolor="#223C2B", line=dict(color="#457053", width=1.5), layer="below")
            fig_mini.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="#5C8A6B", width=1.5))
            fig_mini.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="#5C8A6B", width=1.5))
            fig_mini.add_shape(type="rect", x0=120, y0=36, x1=121.5, y1=44, line=dict(color="#FFFFFF", width=2), fillcolor="#FFFFFF")
            fig_mini.add_shape(type="line", x0=120, y0=10, x1=120, y1=70, line=dict(color="#FFFFFF", width=2))
            fig_mini.add_shape(type="circle", x0=107.5, y0=39.5, x1=108.5, y1=40.5, fillcolor="#FFFFFF", line=dict(color="#FFFFFF"))

            for _, r in p_shots.iterrows():
                out = str(r.get("shot_outcome_name", "") or "")
                xg_raw = pd.to_numeric(r.get("shot_statsbomb_xg"), errors="coerce")
                xg_val = float(xg_raw) if pd.notna(xg_raw) else 0.08
                col = "#00D66C" if out == "Goal" else ("#38BDF8" if out in ["Saved", "Saved to Post"] else ("#94A3B8" if out == "Blocked" else "#F97316"))
                sym = "star" if out == "Goal" else ("circle" if out in ["Saved", "Saved to Post", "Blocked"] else "x")
                sz = 14 if out == "Goal" else 10
                sx = pd.to_numeric(r.get("location_x"), errors="coerce")
                sy = pd.to_numeric(r.get("location_y"), errors="coerce")
                if pd.notna(sx) and pd.notna(sy):
                    sx_f, sy_f = float(sx), float(sy)
                    if sx_f < 60.0:
                        sx_f = 120.0 - sx_f
                        sy_f = 80.0 - sy_f
                    s_type = str(r.get("shot_type_name") or "Open Play")
                    if s_type.lower() == "nan": s_type = "Open Play"
                    s_body = str(r.get("shot_body_part_name") or "Foot")
                    if s_body.lower() == "nan": s_body = "Foot"
                    m_val = int(pd.to_numeric(r.get("minute", 0), errors="coerce") or 0)
                    fig_mini.add_trace(go.Scatter(
                        x=[sx_f], y=[sy_f], mode="markers",
                        marker=dict(size=sz, color=col, symbol=sym, line=dict(color="#FFFFFF", width=1.5)),
                        name=f"{m_val}' - {out}",
                        hovertext=f"Phút {m_val}': {out}<br>xG: {xg_val:.2f}<br>{s_type} • {s_body}"
                    ))

            fig_mini.update_layout(
                xaxis=dict(range=[73, 123], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[8, 72], showgrid=False, zeroline=False, showticklabels=False),
                height=210, margin=dict(l=2, r=2, t=4, b=4),
                plot_bgcolor="#131714", paper_bgcolor="#0D100E", showlegend=False
            )
            st.plotly_chart(fig_mini, width="stretch", key=f"mini_shots_{p_name}_{current_match['id']}")

            shot_items_html = []
            for _, r in p_shots.iterrows():
                out = str(r.get("shot_outcome_name", "") or "Shot")
                xg_raw = pd.to_numeric(r.get("shot_statsbomb_xg"), errors="coerce")
                xg_val = float(xg_raw) if pd.notna(xg_raw) else 0.08
                badge_bg = "#00D66C" if out == "Goal" else ("#38BDF8" if out in ["Saved", "Saved to Post"] else ("#64748B" if out == "Blocked" else "#F97316"))
                s_type = str(r.get("shot_type_name") or "Open Play")
                if s_type.lower() == "nan": s_type = "Open Play"
                s_body = str(r.get("shot_body_part_name") or "Foot")
                if s_body.lower() == "nan": s_body = "Foot"
                m_val = int(pd.to_numeric(r.get("minute", 0), errors="coerce") or 0)
                shot_items_html.append(
                    f'<div style="background: #131A26; border-radius: 8px; padding: 8px 12px; border: 1px solid #1E2638; display: flex; justify-content: space-between; align-items: center;">'
                    f'<div><span style="font-weight: 800; color: #FFFFFF; font-size: 13px;">Phút {m_val}\'</span>'
                    f'<span style="background: {badge_bg}; color: #000; font-weight: 800; font-size: 10.5px; padding: 2px 6px; border-radius: 4px; margin-left: 7px;">{html.escape(out)}</span>'
                    f'<div style="font-size: 11.5px; color: #8E9BAE; margin-top: 2px;">{html.escape(s_type)} • {html.escape(s_body)}</div></div>'
                    f'<div style="text-align: right;"><span style="font-size: 10.5px; color: #8E9BAE;">xG</span><br>'
                    f'<span style="font-weight: 800; color: #38BDF8; font-size: 14px;">{xg_val:.2f}</span></div></div>'
                )
            st.markdown(
                f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 7px; max-height: 215px; overflow-y: auto; padding-right: 4px; margin-top: 4px;">{"".join(shot_items_html)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("ℹ️ Cầu thủ này không thực hiện cú sút nào trong trận đấu.")

    with sub_heat:
        if {'location_x', 'location_y'}.issubset(p_events.columns):
            loc_df = p_events[['location_x', 'location_y']].copy()
            loc_df['location_x'] = pd.to_numeric(loc_df['location_x'], errors='coerce')
            loc_df['location_y'] = pd.to_numeric(loc_df['location_y'], errors='coerce')
            loc_df = loc_df.dropna(subset=['location_x', 'location_y'])
        else:
            loc_df = pd.DataFrame()

        if len(loc_df) > 0:
            fig_hm = go.Figure()
            for i in range(12):
                sc = "#121B14" if i % 2 == 0 else "#152217"
                fig_hm.add_shape(type="rect", x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80, fillcolor=sc, line=dict(width=0), layer="below")
            if len(loc_df) >= 3:
                sofascore_scale = [
                    [0.0, "rgba(18, 27, 20, 0.0)"],
                    [0.12, "rgba(16, 185, 129, 0.25)"],
                    [0.35, "rgba(132, 204, 22, 0.52)"],
                    [0.60, "rgba(250, 204, 21, 0.76)"],
                    [0.82, "rgba(249, 115, 22, 0.88)"],
                    [1.0, "rgba(239, 68, 68, 0.96)"],
                ]
                fig_hm.add_trace(go.Histogram2dContour(
                    x=loc_df['location_x'], y=loc_df['location_y'],
                    colorscale=sofascore_scale, reversescale=False,
                    ncontours=18, showscale=False,
                    contours=dict(coloring="fill", showlines=False),
                    hoverinfo="skip"
                ))
            fig_hm.add_trace(go.Scatter(
                x=loc_df['location_x'], y=loc_df['location_y'],
                mode="markers",
                marker=dict(size=4.5 if len(loc_df) >= 3 else 10, color="rgba(255, 255, 255, 0.42)" if len(loc_df) >= 3 else "#FF9800", line=dict(color="#0B0F19", width=0.5)),
                hovertemplate="Tọa độ: (%{x:.1f}, %{y:.1f})<extra></extra>", showlegend=False
            ))
            fig_hm.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="rgba(255,255,255,0.45)", width=1.6), layer="above")
            fig_hm.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="rgba(255,255,255,0.45)", width=1.4), layer="above")
            fig_hm.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="rgba(255,255,255,0.45)", width=1.4), layer="above")
            fig_hm.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="rgba(255,255,255,0.45)", width=1.4), layer="above")
            fig_hm.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="rgba(255,255,255,0.45)", width=1.4), layer="above")
            fig_hm.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="rgba(255,255,255,0.45)", width=1.2), layer="above")
            fig_hm.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="rgba(255,255,255,0.45)", width=1.2), layer="above")
            fig_hm.add_annotation(
                x=60, y=76.2, text=f"HƯỚNG TẤN CÔNG CỦA {team_name.upper()} ➔",
                showarrow=False, font=dict(size=10, color="#A7F3D0", family="Arial Black, sans-serif"),
                bgcolor="rgba(15, 23, 42, 0.78)", borderpad=3
            )
            fig_hm.update_layout(
                xaxis=dict(range=[-2, 122], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                yaxis=dict(range=[82, -2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True),
                height=310, margin=dict(l=4, r=4, t=4, b=4),
                plot_bgcolor="#121B14", paper_bgcolor="#0D100E", showlegend=False
            )
            st.plotly_chart(fig_hm, width="stretch", key=f"heat_{p_name}_{current_match['id']}")
        else:
            st.info("ℹ️ Chưa có đủ dữ liệu tọa độ sự kiện để vẽ bản đồ nhiệt cho cầu thủ này.")

    with sub_radar:
        render_player_radar_and_fifa_subtab(player_data, p_events, current_match)


# ==========================================================
# 🗺️ TỌA ĐỘ VỊ TRÍ CHIẾN THUẬT BÓNG ĐÁ TRÊN SÂN (ĐẦY ĐỦ 25 VỊ TRÍ STATSBOMB)
# ==========================================================
POSITION_COORDS = {
    "home": {
        "Goalkeeper": (8, 40), "GK": (8, 40),
        "Right Back": (23, 15), "RB": (23, 15),
        "Right Wing Back": (29, 15), "RWB": (29, 15),
        "Right Center Back": (20, 28), "RCB": (20, 28),
        "Center Back": (19, 40), "CB": (19, 40),
        "Left Center Back": (20, 52), "LCB": (20, 52),
        "Left Back": (23, 63), "LB": (23, 63),
        "Left Wing Back": (29, 63), "LWB": (29, 63),
        "Right Defensive Midfield": (33, 27), "RDM": (33, 27),
        "Center Defensive Midfield": (32, 40), "CDM": (32, 40), "DM": (32, 40),
        "Left Defensive Midfield": (33, 53), "LDM": (33, 53),
        "Right Midfield": (42, 15), "RM": (42, 15),
        "Right Center Midfield": (41, 28), "RCM": (41, 28),
        "Center Midfield": (40, 40), "CM": (40, 40),
        "Left Center Midfield": (41, 52), "LCM": (41, 52),
        "Left Midfield": (42, 63), "LM": (42, 63),
        "Right Attacking Midfield": (47, 25), "RAM": (47, 25),
        "Center Attacking Midfield": (47, 40), "CAM": (47, 40), "AM": (47, 40),
        "Left Attacking Midfield": (47, 55), "LAM": (47, 55),
        "Right Wing": (52, 16), "RW": (52, 16),
        "Left Wing": (52, 62), "LW": (52, 62),
        "Second Striker": (50, 40), "Secondary Striker": (50, 40), "SS": (50, 40),
        "Right Center Forward": (54, 29), "RCF": (54, 29),
        "Center Forward": (55, 40), "CF": (55, 40), "ST": (55, 40),
        "Left Center Forward": (54, 51), "LCF": (54, 51),
    },
    "away": {
        "Goalkeeper": (112, 40), "GK": (112, 40),
        "Right Back": (97, 15), "RB": (97, 15),
        "Right Wing Back": (91, 15), "RWB": (91, 15),
        "Right Center Back": (100, 28), "RCB": (100, 28),
        "Center Back": (101, 40), "CB": (101, 40),
        "Left Center Back": (100, 52), "LCB": (100, 52),
        "Left Back": (97, 63), "LB": (97, 63),
        "Left Wing Back": (91, 63), "LWB": (91, 63),
        "Right Defensive Midfield": (87, 27), "RDM": (87, 27),
        "Center Defensive Midfield": (88, 40), "CDM": (88, 40), "DM": (88, 40),
        "Left Defensive Midfield": (87, 53), "LDM": (87, 53),
        "Right Midfield": (78, 15), "RM": (78, 15),
        "Right Center Midfield": (79, 28), "RCM": (79, 28),
        "Center Midfield": (80, 40), "CM": (80, 40),
        "Left Center Midfield": (79, 52), "LCM": (79, 52),
        "Left Midfield": (78, 63), "LM": (78, 63),
        "Right Attacking Midfield": (73, 25), "RAM": (73, 25),
        "Center Attacking Midfield": (73, 40), "CAM": (73, 40), "AM": (73, 40),
        "Left Attacking Midfield": (73, 55), "LAM": (73, 55),
        "Right Wing": (68, 16), "RW": (68, 16),
        "Left Wing": (68, 62), "LW": (68, 62),
        "Second Striker": (70, 40), "Secondary Striker": (70, 40), "SS": (70, 40),
        "Right Center Forward": (66, 29), "RCF": (66, 29),
        "Center Forward": (65, 40), "CF": (65, 40), "ST": (65, 40),
        "Left Center Forward": (66, 51), "LCF": (66, 51),
    }
}

def get_tactical_coords(pos_name, idx, is_home):
    side = "home" if is_home else "away"
    coords = POSITION_COORDS[side].get(str(pos_name or "").strip())
    if coords:
        return float(coords[0]), float(coords[1])
    if idx == 0:
        return (10.0, 40.0) if is_home else (110.0, 40.0)
    elif idx in [1, 2, 3, 4]:
        return (24.0, 14.0 + (idx - 1) * 17.0) if is_home else (96.0, 14.0 + (idx - 1) * 17.0)
    elif idx in [5, 6, 7]:
        return (39.0, 20.0 + (idx - 5) * 20.0) if is_home else (81.0, 20.0 + (idx - 5) * 20.0)
    else:
        return (53.0, 20.0 + (idx - 8) * 20.0) if is_home else (67.0, 20.0 + (idx - 8) * 20.0)


# ==========================================================
# 📦 BỘ PHÂN TÍCH & TRÍCH XUẤT SỰ KIỆN ĐỘNG (DYNAMIC ENGINE)
# ==========================================================

def parse_statsbomb_events_json(raw_events, match_id=None):
    valid_periods = {1, 2, 3, 4}
    starting_xi = [e for e in raw_events if (e.get('type') or {}).get('name') == 'Starting XI']
    starting_lineups = {}
    team_names = []
    
    for e in starting_xi:
        t_name = (e.get('team') or {}).get('name')
        if t_name and t_name not in team_names:
            team_names.append(t_name)
        tactics = e.get('tactics') or {}
        formation = str(tactics.get('formation', ''))
        players = []
        for p in tactics.get('lineup', []):
            players.append({
                'name': (p.get('player') or {}).get('name', ''),
                'position': (p.get('position') or {}).get('name', ''),
                'jersey_number': p.get('jersey_number', '')
            })
        starting_lineups[t_name] = {'formation': formation, 'players': players}
        
    if len(team_names) < 2:
        all_teams = list(set((e.get('team') or {}).get('name') for e in raw_events if (e.get('team') or {}).get('name')))
        for t in all_teams:
            if t not in team_names:
                team_names.append(t)
    
    home_team = team_names[0] if len(team_names) > 0 else "Home Team"
    away_team = team_names[1] if len(team_names) > 1 else "Away Team"
    
    rows = []
    goals_info = []
    home_score = 0
    away_score = 0
    pen_home = 0
    pen_away = 0
    has_pen_shootout = False

    for e in raw_events:
        period = e.get('period', 1)
        t_name = (e.get('type') or {}).get('name', '')
        team = (e.get('team') or {}).get('name', '')
        if period == 5:
            if t_name == 'Shot':
                has_pen_shootout = True
                if ((e.get('shot') or {}).get('outcome') or {}).get('name') == 'Goal':
                    if team == home_team:
                        pen_home += 1
                    elif team == away_team:
                        pen_away += 1
            continue
        if period not in valid_periods:
            continue

        player = (e.get('player') or {}).get('name', '')
        pos = (e.get('position') or {}).get('name', '')
        minute = e.get('minute', 0)
        second = e.get('second', 0)
        timestamp = e.get('timestamp', '00:00:00')

        loc = e.get('location') or []
        x = loc[0] if len(loc) > 0 else None
        y = loc[1] if len(loc) > 1 else None

        pass_data = e.get('pass') or {}
        p_end = pass_data.get('end_location') or []
        px = p_end[0] if len(p_end) > 0 else None
        py = p_end[1] if len(p_end) > 1 else None
        p_rec = (pass_data.get('recipient') or {}).get('name', '')
        p_outcome = (pass_data.get('outcome') or {}).get('name', '')

        shot_data = e.get('shot') or {}
        s_outcome = (shot_data.get('outcome') or {}).get('name', '')
        s_xg = shot_data.get('statsbomb_xg')
        if t_name == 'Shot' and (s_xg is None or pd.isna(s_xg)) and x is not None and y is not None:
            s_xg = estimate_xg(x, y)
        shot_type = (shot_data.get('type') or {}).get('name', 'Open Play') if t_name == 'Shot' else ''
        shot_body = (shot_data.get('body_part') or {}).get('name', 'Foot') if t_name == 'Shot' else ''
        dribble_outcome = (e.get('dribble') or {}).get('outcome', {}).get('name', '')
        foul_card = ((e.get('foul_committed') or {}).get('card') or (e.get('bad_behaviour') or {}).get('card') or {}).get('name', '')
        substitution_data = e.get('substitution') or {}
        substitution_replacement = substitution_data.get('replacement') or {}
        substitution_outcome = substitution_data.get('outcome') or {}
        gk_data = e.get('goalkeeper') or {}
        gk_type_name = (gk_data.get('type') or {}).get('name', '')
        gk_outcome_name = (gk_data.get('outcome') or {}).get('name', '')
        carry_end = (e.get('carry') or {}).get('end_location') or []
        carry_dist = 0.0
        if t_name == 'Carry' and x is not None and y is not None and len(carry_end) >= 2:
            carry_dist = round(float(np.sqrt((carry_end[0] - x)**2 + (carry_end[1] - y)**2)), 1)

        if t_name == 'Shot' and s_outcome == 'Goal' and period in valid_periods:
            goals_info.append({"player": player, "minute": minute, "team": team, "is_og": False})
            if team == home_team:
                home_score += 1
            elif team == away_team:
                away_score += 1
        elif t_name == 'Own Goal Against' and period in valid_periods:
            benefiting_team = away_team if team == home_team else home_team
            goals_info.append({"player": player, "minute": minute, "team": benefiting_team, "is_og": True})
            if benefiting_team == home_team:
                home_score += 1
            elif benefiting_team == away_team:
                away_score += 1

        row = {
            'match_id': match_id or 999999,
            'minute': minute,
            'second': second,
            'period': period,
            'timestamp': timestamp,
            'team_name': team,
            'player_name': player,
            'position_name': pos,
            'type_name': t_name,
            'location_x': x,
            'location_y': y,
            'pass_recipient_name': p_rec,
            'pass_outcome_name': p_outcome,
            'pass_end_location_x': px,
            'pass_end_location_y': py,
            'pass_length': pass_data.get('length'),
            'pass_cross': 1 if pass_data.get('cross') else 0,
            'pass_through_ball': 1 if pass_data.get('through_ball') else 0,
            'pass_switch': 1 if pass_data.get('switch') else 0,
            'pass_shot_assist': 1 if pass_data.get('shot_assist') else 0,
            'pass_goal_assist': 1 if pass_data.get('goal_assist') else 0,
            'shot_outcome_name': s_outcome,
            'shot_statsbomb_xg': s_xg,
            'shot_type_name': shot_type,
            'shot_body_part_name': shot_body,
            'dribble_outcome_name': dribble_outcome,
            'card_name': foul_card,
            'substitution_replacement_name': substitution_replacement.get('name', ''),
            'substitution_outcome_name': substitution_outcome.get('name', ''),
            'gk_type_name': gk_type_name,
            'gk_outcome_name': gk_outcome_name,
            'carry_distance': carry_dist,
            'under_pressure': 1 if e.get('under_pressure') else 0,
            'possession': e.get('possession'),
            'possession_team': (e.get('possession_team') or {}).get('name', ''),
        }
        rows.append(row)

    events_df = pd.DataFrame(rows)

    # Tính thống kê trận đấu
    stats_dict = {t: {
        'possession_seconds': 0.0, 'shots': 0, 'shots_on_target': 0, 'shots_off_target': 0,
        'blocked_shots': 0, 'xg': 0.0, 'key_passes': 0, 'goalkeeper_saves': 0, 'corners': 0,
        'fouls': 0, 'free_kicks': 0, 'yellow_cards': 0, 'red_cards': 0, 'goals': 0,
        'passes': 0, 'passes_completed': 0
    } for t in [home_team, away_team]}

    for idx, e in enumerate(raw_events[:-1]):
        ne = raw_events[idx + 1]
        p = e.get('period')
        p_team = (e.get('possession_team') or {}).get('name')
        if (p in valid_periods and ne.get('period') == p and 
            e.get('possession') == ne.get('possession') and 
            p_team in stats_dict and (ne.get('possession_team') or {}).get('name') == p_team):
            gap = _sec(ne.get('timestamp', '00:00:00')) - _sec(e.get('timestamp', '00:00:00'))
            if 0 < gap <= 10.0:
                stats_dict[p_team]['possession_seconds'] += gap

    tot_pos_sec = sum(stats_dict[t]['possession_seconds'] for t in stats_dict)
    for t in stats_dict:
        stats_dict[t]['possession_pct'] = round(stats_dict[t]['possession_seconds'] / tot_pos_sec * 100, 1) if tot_pos_sec > 0 else 50.0

    for e in raw_events:
        if e.get('period') not in valid_periods:
            continue
        t_name = (e.get('type') or {}).get('name')
        team = (e.get('team') or {}).get('name')
        if team not in stats_dict:
            continue

        if t_name == 'Pass':
            stats_dict[team]['passes'] += 1
            p_data = e.get('pass') or {}
            if not p_data.get('outcome'):
                stats_dict[team]['passes_completed'] += 1
            if p_data.get('shot_assist'):
                stats_dict[team]['key_passes'] += 1
            if (p_data.get('type') or {}).get('name') == 'Corner':
                stats_dict[team]['corners'] += 1
            elif (p_data.get('type') or {}).get('name') == 'Free Kick':
                stats_dict[team]['free_kicks'] += 1

        elif t_name == 'Shot':
            stats_dict[team]['shots'] += 1
            s_data = e.get('shot') or {}
            s_xg = float(s_data.get('statsbomb_xg') or 0.0)
            stats_dict[team]['xg'] += s_xg
            outcome = (s_data.get('outcome') or {}).get('name')
            if outcome in {'Goal', 'Saved', 'Saved to Post'}:
                stats_dict[team]['shots_on_target'] += 1
            elif outcome in {'Off T', 'Wayward', 'Post', 'Saved Off T'}:
                stats_dict[team]['shots_off_target'] += 1
            elif outcome == 'Blocked':
                stats_dict[team]['blocked_shots'] += 1
            if outcome == 'Goal':
                stats_dict[team]['goals'] += 1
            if outcome in {'Saved', 'Saved to Post'}:
                def_team = away_team if team == home_team else home_team
                stats_dict[def_team]['goalkeeper_saves'] += 1

        elif t_name == 'Own Goal Against':
            def_team = away_team if team == home_team else home_team
            stats_dict[def_team]['goals'] += 1

        elif t_name == 'Foul Committed':
            stats_dict[team]['fouls'] += 1
            card = ((e.get('foul_committed') or {}).get('card') or {}).get('name')
            if card in {'Yellow Card', 'Second Yellow'}:
                stats_dict[team]['yellow_cards'] += 1
            if card in {'Red Card', 'Second Yellow'}:
                stats_dict[team]['red_cards'] += 1

        elif t_name == 'Bad Behaviour':
            card = ((e.get('bad_behaviour') or {}).get('card') or {}).get('name')
            if card in {'Yellow Card', 'Second Yellow'}:
                stats_dict[team]['yellow_cards'] += 1
            if card in {'Red Card', 'Second Yellow'}:
                stats_dict[team]['red_cards'] += 1

    for t in stats_dict:
        stats_dict[t]['xg'] = round(stats_dict[t]['xg'], 2)

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_score': home_score,
        'away_score': away_score,
        'pen_home': pen_home if has_pen_shootout else None,
        'pen_away': pen_away if has_pen_shootout else None,
        'goals': goals_info,
        'events_df': events_df,
        'stats_dict': stats_dict,
        'starting_lineups': starting_lineups
    }


def parse_statsbomb_lineups_json(raw_lineups):
    """Read StatsBomb's full match squad, including substitutes who never played."""
    squads = {}
    for team_data in raw_lineups or []:
        team_name = str(team_data.get('team_name') or '').strip()
        if not team_name:
            continue

        players = []
        for player in team_data.get('lineup') or []:
            positions = player.get('positions') or []
            reasons_on = [str(pos.get('start_reason') or '').casefold() for pos in positions]
            reasons_off = [str(pos.get('end_reason') or '').casefold() for pos in positions]
            started = any('starting xi' in reason for reason in reasons_on)
            entered = any('substitution - on' in reason for reason in reasons_on)
            left = any('substitution - off' in reason for reason in reasons_off)
            injured_on_exit = any('substitution - off (injury)' in reason for reason in reasons_off)
            first_position = next((pos for pos in positions if pos.get('position')), {})
            players.append({
                'name': str(player.get('player_name') or '').strip(),
                'nickname': str(player.get('player_nickname') or '').strip(),
                'jersey_number': player.get('jersey_number'),
                'position': str(first_position.get('position') or '').strip(),
                'started': started,
                'entered': entered,
                'left': left,
                'injured_on_exit': injured_on_exit,
                'minute_in': next((pos.get('from') for pos in positions if 'substitution - on' in str(pos.get('start_reason') or '').casefold()), None),
                'minute_out': next((pos.get('to') for pos in positions if 'substitution - off' in str(pos.get('end_reason') or '').casefold()), None),
            })

        squads[team_name] = {'players': players}
    return squads

def parse_uploaded_csv(df_raw):
    col_map = {}
    for c in df_raw.columns:
        cl = str(c).strip().lower()
        if cl in ['team', 'team_name', 'teamname']: col_map[c] = 'team_name'
        elif cl in ['player', 'player_name', 'playername']: col_map[c] = 'player_name'
        elif cl in ['type', 'type_name', 'typename', 'event_type', 'eventtype']: col_map[c] = 'type_name'
        elif cl in ['x', 'location_x', 'start_x', 'startx', 'x1']: col_map[c] = 'location_x'
        elif cl in ['y', 'location_y', 'start_y', 'starty', 'y1']: col_map[c] = 'location_y'
        elif cl in ['end_x', 'pass_end_location_x', 'endx', 'x2']: col_map[c] = 'pass_end_location_x'
        elif cl in ['end_y', 'pass_end_location_y', 'endy', 'y2']: col_map[c] = 'pass_end_location_y'
        elif cl in ['recipient', 'pass_recipient_name', 'recipient_name']: col_map[c] = 'pass_recipient_name'
        elif cl in ['outcome', 'pass_outcome_name', 'outcome_name']: col_map[c] = 'pass_outcome_name'
        elif cl in ['shot_outcome', 'shot_outcome_name']: col_map[c] = 'shot_outcome_name'
        elif cl in ['xg', 'shot_statsbomb_xg', 'expected_goals']: col_map[c] = 'shot_statsbomb_xg'
        elif cl in ['replacement', 'replacement_name', 'substitution_replacement', 'substitution_replacement_name', 'player_in', 'incoming_player']: col_map[c] = 'substitution_replacement_name'
        elif cl in ['substitution_outcome', 'substitution_outcome_name', 'substitution_reason']: col_map[c] = 'substitution_outcome_name'
        elif cl in ['minute', 'min']: col_map[c] = 'minute'
        elif cl in ['position', 'position_name', 'pos']: col_map[c] = 'position_name'
            
    df = df_raw.rename(columns=col_map).copy()
    if 'period' in df.columns:
        numeric_period = pd.to_numeric(df['period'], errors='coerce')
        df = df.loc[numeric_period.isna() | numeric_period.isin([1, 2, 3, 4])].copy()
    if 'type_name' not in df.columns:
        df['type_name'] = np.where(df.get('pass_end_location_x', pd.Series()).notna(), 'Pass', 'Event')
        
    if 'shot_outcome_name' not in df.columns and 'pass_outcome_name' in df.columns:
        df['shot_outcome_name'] = np.where(df['type_name'] == 'Shot', df['pass_outcome_name'], None)
            
    teams = [t for t in df['team_name'].dropna().unique().tolist() if str(t) != 'nan'] if 'team_name' in df.columns else []
    home_team = teams[0] if len(teams) > 0 else "Home Team"
    away_team = teams[1] if len(teams) > 1 else "Away Team"
    
    goals_info = []
    home_score = 0
    away_score = 0
    if 'shot_outcome_name' in df.columns:
        shots = df[df['shot_outcome_name'].astype(str).str.lower().isin(['goal', '1', 'true'])]
        for _, s_row in shots.iterrows():
            p_name = str(s_row.get('player_name', 'Player'))
            p_min = int(s_row.get('minute', 0)) if pd.notnull(s_row.get('minute')) else 0
            p_team = str(s_row.get('team_name', ''))
            goals_info.append({"player": p_name, "minute": p_min, "team": p_team})
            if p_team == home_team: home_score += 1
            elif p_team == away_team: away_score += 1
                
    if 'shot_statsbomb_xg' not in df.columns and 'location_x' in df.columns and 'location_y' in df.columns:
        df['shot_statsbomb_xg'] = df.apply(lambda r: estimate_xg(r['location_x'], r['location_y']) if r.get('type_name') == 'Shot' else 0.0, axis=1)
        
    stats_dict = {t: {
        'possession_pct': 50.0, 'shots': 0, 'shots_on_target': 0, 'shots_off_target': 0,
        'blocked_shots': 0, 'xg': 0.0, 'key_passes': 0, 'goalkeeper_saves': 0, 'corners': 0,
        'fouls': 0, 'free_kicks': 0, 'yellow_cards': 0, 'red_cards': 0, 'goals': 0,
        'passes': 0, 'passes_completed': 0
    } for t in [home_team, away_team]}
    
    for t in [home_team, away_team]:
        t_ev = df[df['team_name'] == t]
        passes = t_ev[t_ev['type_name'] == 'Pass']
        stats_dict[t]['passes'] = len(passes)
        comp_mask = _pass_success_mask(passes)
        stats_dict[t]['passes_completed'] = int(comp_mask.sum()) if comp_mask is not None else len(passes)
        shots = t_ev[t_ev['type_name'] == 'Shot']
        stats_dict[t]['shots'] = len(shots)
        if 'shot_statsbomb_xg' in t_ev.columns:
            stats_dict[t]['xg'] = round(float(pd.to_numeric(t_ev['shot_statsbomb_xg'], errors='coerce').fillna(0).sum()), 2)
        if 'shot_outcome_name' in shots.columns:
            stats_dict[t]['shots_on_target'] = len(shots[shots['shot_outcome_name'].isin(['Goal', 'Saved'])])
            stats_dict[t]['shots_off_target'] = len(shots[shots['shot_outcome_name'].isin(['Off T', 'Wayward', 'Post'])])
            stats_dict[t]['blocked_shots'] = len(shots[shots['shot_outcome_name'] == 'Blocked'])
            stats_dict[t]['goals'] = len(shots[shots['shot_outcome_name'] == 'Goal'])
            
    tot_p = stats_dict[home_team]['passes'] + stats_dict[away_team]['passes']
    if tot_p > 0:
        stats_dict[home_team]['possession_pct'] = round(stats_dict[home_team]['passes'] / tot_p * 100, 1)
        stats_dict[away_team]['possession_pct'] = round(100.0 - stats_dict[home_team]['possession_pct'], 1)
        
    return {
        'home_team': home_team, 'away_team': away_team,
        'home_score': home_score, 'away_score': away_score,
        'goals': goals_info, 'events_df': df,
        'stats_dict': stats_dict, 'starting_lineups': {}
    }


# ==========================================================
# 📊 FOTMOB / SOFASCORE COMPARISON STAT BARS
# ==========================================================

def render_fotmob_stat_bar(label, home_val, away_val, home_str, away_str, home_color="#38BDF8", away_color="#FF4D6D"):
    h_lead, a_lead = False, False
    try:
        h_clean = str(home_val).replace(",", "").replace("%", "").replace("+", "").strip()
        a_clean = str(away_val).replace(",", "").replace("%", "").replace("+", "").strip()
        h_num = float(h_clean)
        a_num = float(a_clean)
        total = abs(h_num) + abs(a_num)
        if total > 0:
            h_pct = max(8.0, min(92.0, round(abs(h_num) / total * 100, 1)))
            a_pct = round(100.0 - h_pct, 1)
        else:
            h_pct, a_pct = 50.0, 50.0
        h_lead = h_num > a_num
        a_lead = a_num > h_num
    except Exception:
        h_pct, a_pct = 50.0, 50.0

    h_badge_style = (
        f"background: rgba(56, 189, 248, 0.22); border: 1px solid {home_color}; color: #FFFFFF; padding: 2px 10px; border-radius: 999px; font-weight: 900; font-size: 14.5px; box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);"
        if h_lead else
        "color: #E2E8F0; font-weight: 700; font-size: 14.5px; padding: 2px 4px;"
    )
    a_badge_style = (
        f"background: rgba(255, 77, 109, 0.22); border: 1px solid {away_color}; color: #FFFFFF; padding: 2px 10px; border-radius: 999px; font-weight: 900; font-size: 14.5px; box-shadow: 0 0 10px rgba(255, 77, 109, 0.25);"
        if a_lead else
        "color: #E2E8F0; font-weight: 700; font-size: 14.5px; padding: 2px 4px;"
    )
    h_bar_op = "1.0" if h_lead or (not h_lead and not a_lead) else "0.55"
    a_bar_op = "1.0" if a_lead or (not h_lead and not a_lead) else "0.55"

    return (
        f'<div style="margin-bottom: 12px; background: linear-gradient(180deg, #151E2E 0%, #111827 100%); padding: 12px 18px; border-radius: 12px; border: 1px solid #26354D;">'
        f'<div style="display: grid; grid-template-columns: minmax(90px, 1fr) auto minmax(90px, 1fr); align-items: center; gap: 10px; margin-bottom: 8px;">'
        f'<div style="text-align: left;"><span style="{h_badge_style}">{home_str}</span></div>'
        f'<div style="color: #F8FAFC; font-weight: 800; font-size: 13.5px; text-transform: uppercase; letter-spacing: 0.6px; text-align: center;">{label}</div>'
        f'<div style="text-align: right;"><span style="{a_badge_style}">{away_str}</span></div>'
        f'</div>'
        f'<div style="display: flex; height: 9px; border-radius: 999px; overflow: hidden; background-color: #1E293B; gap: 3px;">'
        f'<div style="width: {h_pct}%; background: linear-gradient(90deg, #0284C7 0%, {home_color} 100%); opacity: {h_bar_op}; border-radius: 999px 0 0 999px;"></div>'
        f'<div style="width: {a_pct}%; background: linear-gradient(90deg, {away_color} 0%, #E11D48 100%); opacity: {a_bar_op}; border-radius: 0 999px 999px 0;"></div>'
        f'</div>'
        f'</div>'
    )


# ==========================================================
# 🎯 BẢN ĐỒ DỨT ĐIỂM (SHOT MAP)
# ==========================================================

def render_shot_map(match_events, home_team, away_team, filter_choice="Tất cả cú sút", current_match=None):
    shots = match_events[match_events['type_name'] == 'Shot'].copy() if 'type_name' in match_events.columns else pd.DataFrame()

    if 'period' in shots.columns:
        numeric_period = pd.to_numeric(shots['period'], errors='coerce')
        shots = shots.loc[numeric_period.isna() | numeric_period.isin([1, 2, 3, 4])].copy()

    if shots.empty:
        if current_match and current_match.get('goals'):
            st.info("ℹ️ Có dữ liệu bàn thắng nhưng nguồn không có tọa độ cú sút để dựng bản đồ chính xác.")
        else:
            st.info("ℹ️ Không tìm thấy dữ liệu cú sút trong trận đấu này.")
        return

    # CSV uploads may omit optional shot fields. Keep the map usable without
    # treating a missing column as a scalar value or crashing on .fillna().
    if 'team_name' not in shots.columns:
        shots['team_name'] = ''
    if 'player_name' not in shots.columns:
        shots['player_name'] = 'Cầu thủ'
    if 'minute' not in shots.columns:
        shots['minute'] = 0
    if 'shot_outcome_name' not in shots.columns:
        shots['shot_outcome_name'] = ''
        
    shots['location_x'] = pd.to_numeric(shots['location_x'], errors='coerce')
    shots['location_y'] = pd.to_numeric(shots['location_y'], errors='coerce')
    xg_values = (
        pd.to_numeric(shots['shot_statsbomb_xg'], errors='coerce')
        if 'shot_statsbomb_xg' in shots.columns
        else pd.Series(index=shots.index, dtype='float64')
    )
    shots['shot_statsbomb_xg'] = xg_values.fillna(0.08)
    shots = shots.dropna(subset=['location_x', 'location_y'])
    
    if shots.empty:
        st.info("ℹ️ Tọa độ các cú sút trong sự kiện không hợp lệ.")
        return
        
    # Tính toán các chỉ số so sánh (KPI Cards) cho 2 đội
    h_shots_all = shots[shots['team_name'] == home_team]
    a_shots_all = shots[shots['team_name'] == away_team]
    
    h_tot = len(h_shots_all)
    a_tot = len(a_shots_all)
    
    on_target_outcomes = {'goal', 'saved', 'saved to post'}
    h_outcomes = h_shots_all['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold()
    a_outcomes = a_shots_all['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold()
    h_on = int(h_outcomes.isin(on_target_outcomes).sum())
    a_on = int(a_outcomes.isin(on_target_outcomes).sum())
    
    h_xg = float(h_shots_all['shot_statsbomb_xg'].sum())
    a_xg = float(a_shots_all['shot_statsbomb_xg'].sum())
    
    h_goals = int(h_outcomes.eq('goal').sum())
    a_goals = int(a_outcomes.eq('goal').sum())
    
    h_acc_str = f"{h_on}/{h_tot} ({h_on/h_tot*100:.0f}%)" if h_tot > 0 else "0/0 (0%)"
    a_acc_str = f"{a_on}/{a_tot} ({a_on/a_tot*100:.0f}%)" if a_tot > 0 else "0/0 (0%)"

    def _mini_kpi_card(title, h_text, a_text, h_val_num, a_val_num):
        tot_v = float(h_val_num) + float(a_val_num)
        hp = round(float(h_val_num) / tot_v * 100, 1) if tot_v > 0 else 50.0
        ap = round(100.0 - hp, 1)
        return (
            f'<div style="background: linear-gradient(180deg, #151E2E 0%, #111827 100%); padding: 14px 16px; border-radius: 12px; border: 1px solid #26354D; text-align: center;">'
            f'<div style="color: #94A3B8; font-size: 11.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;">{title}</div>'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0 8px;">'
            f'<span style="color: #38BDF8; font-size: 19px; font-weight: 900;">{h_text}</span>'
            f'<span style="color: #64748B; font-size: 12px; font-weight: 700;">VS</span>'
            f'<span style="color: #FF4D6D; font-size: 19px; font-weight: 900;">{a_text}</span>'
            f'</div>'
            f'<div style="display: flex; height: 6px; border-radius: 999px; overflow: hidden; background: #1E293B; gap: 2px;">'
            f'<div style="width: {hp}%; background: #38BDF8;"></div>'
            f'<div style="width: {ap}%; background: #FF4D6D;"></div>'
            f'</div>'
            f'</div>'
        )

    kpi_html = (
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px;">'
        + _mini_kpi_card("Tổng số cú sút", str(h_tot), str(a_tot), h_tot, a_tot)
        + _mini_kpi_card("Sút trúng đích", h_acc_str, a_acc_str, h_on, a_on)
        + _mini_kpi_card("Bàn thắng kỳ vọng (xG)", f"{h_xg:.2f}", f"{a_xg:.2f}", h_xg, a_xg)
        + _mini_kpi_card("Bàn thắng (Goals)", str(h_goals), str(a_goals), h_goals, a_goals)
        + '</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Lọc danh sách cú sút theo lựa chọn của người dùng
    filtered_shots = shots.copy()
    if filter_choice.startswith(f"🔵 {home_team}"):
        filtered_shots = filtered_shots[filtered_shots['team_name'] == home_team]
    elif filter_choice.startswith(f"🔴 {away_team}"):
        filtered_shots = filtered_shots[filtered_shots['team_name'] == away_team]
    elif "Chỉ bàn thắng" in filter_choice:
        outcome = filtered_shots['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold()
        filtered_shots = filtered_shots[outcome.eq('goal')]
    elif "Trúng đích" in filter_choice:
        outcome = filtered_shots['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold()
        filtered_shots = filtered_shots[outcome.isin(on_target_outcomes)]

    # Tạo bản đồ Plotly sân bóng chuyên nghiệp
    fig_shot = go.Figure()

    # Thảm cỏ sọc sân vận động (12 sọc xen kẽ chân thực)
    for i in range(12):
        stripe_color = '#121B14' if i % 2 == 0 else '#152217'
        fig_shot.add_shape(
            type='rect', x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80,
            fillcolor=stripe_color, line=dict(width=0), layer='below'
        )

    pitch_line = dict(color="rgba(255, 255, 255, 0.45)", width=1.6)

    # Đường biên & Vòng tròn trung tâm
    fig_shot.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=pitch_line)
    fig_shot.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=pitch_line)
    fig_shot.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=pitch_line)
    fig_shot.add_shape(type="circle", x0=59.4, y0=39.4, x1=60.6, y1=40.6, fillcolor="rgba(255,255,255,0.65)", line=dict(width=0))

    # Khu vực cấm địa & Khung thành bên trái (x=0 - nơi Đội khách tấn công vào)
    fig_shot.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=pitch_line)
    fig_shot.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=pitch_line)
    fig_shot.add_shape(type="circle", x0=11.4, y0=39.4, x1=12.6, y1=40.6, fillcolor="rgba(255,255,255,0.65)", line=dict(width=0))
    fig_shot.add_shape(type="rect", x0=-2.8, y0=36, x1=0, y1=44, line=dict(color="#FF4D6D", width=2.5), fillcolor="rgba(255, 77, 109, 0.18)")

    # Khu vực cấm địa & Khung thành bên phải (x=120 - nơi Đội nhà tấn công vào)
    fig_shot.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=pitch_line)
    fig_shot.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=pitch_line)
    fig_shot.add_shape(type="circle", x0=107.4, y0=39.4, x1=108.6, y1=40.6, fillcolor="rgba(255,255,255,0.65)", line=dict(width=0))
    fig_shot.add_shape(type="rect", x0=120, y0=36, x1=122.8, y1=44, line=dict(color="#38BDF8", width=2.5), fillcolor="rgba(56, 189, 248, 0.18)")

    # Nhãn hướng tấn công trực tiếp trên sân bóng (chuẩn FotMob)
    fig_shot.add_annotation(
        x=30, y=76.2, text=f"⬅ 🔴 {away_team.upper()} DỨT ĐIỂM ({a_tot} sút • {a_xg:.2f} xG)",
        showarrow=False, font=dict(size=11.5, color="#FDA4AF", family="Arial Black, sans-serif"),
        bgcolor="rgba(15, 23, 42, 0.78)", bordercolor="rgba(255, 77, 109, 0.45)", borderwidth=1, borderpad=4
    )
    fig_shot.add_annotation(
        x=90, y=76.2, text=f"🔵 {home_team.upper()} DỨT ĐIỂM ({h_tot} sút • {h_xg:.2f} xG) ➔",
        showarrow=False, font=dict(size=11.5, color="#7DD3FC", family="Arial Black, sans-serif"),
        bgcolor="rgba(15, 23, 42, 0.78)", bordercolor="rgba(56, 189, 248, 0.45)", borderwidth=1, borderpad=4
    )

    # Chiều tấn công: Đội nhà tấn công sang phải (0 -> 120), Đội khách tấn công sang trái (120 -> 0)
    home_shots = filtered_shots[filtered_shots['team_name'] == home_team].copy()
    away_shots = filtered_shots[filtered_shots['team_name'] == away_team].copy()

    home_shots['plot_x'] = home_shots['location_x']
    home_shots['plot_y'] = home_shots['location_y']
    away_shots['plot_x'] = 120.0 - away_shots['location_x']
    away_shots['plot_y'] = 80.0 - away_shots['location_y']

    all_shots_plot = pd.concat([home_shots, away_shots], ignore_index=True)

    def _shot_category(outcome):
        o = str(outcome).strip().casefold()
        if o == 'goal':
            return "Goal", "⚽ Bàn thắng", "#00E676", "star", 22
        elif o in {'saved', 'saved to post'}:
            return "Saved", "🧤 Thủ môn cản phá", "#38BDF8", "circle", 15
        elif o == 'blocked':
            return "Blocked", "🛡️ Bị hậu vệ chặn", "#FBBF24", "diamond", 13
        else:
            return "Off_Target", "❌ Ra ngoài / Cột xà", "#FF4D6D", "x", 12

    if not all_shots_plot.empty:
        all_shots_plot['cat_tuple'] = all_shots_plot['shot_outcome_name'].apply(_shot_category)
        all_shots_plot['cat_key'] = all_shots_plot['cat_tuple'].apply(lambda t: t[0])
        all_shots_plot['cat_name'] = all_shots_plot['cat_tuple'].apply(lambda t: t[1])
        all_shots_plot['cat_color'] = all_shots_plot['cat_tuple'].apply(lambda t: t[2])
        all_shots_plot['cat_symbol'] = all_shots_plot['cat_tuple'].apply(lambda t: t[3])
        all_shots_plot['cat_base_sz'] = all_shots_plot['cat_tuple'].apply(lambda t: t[4])

        cat_order = ["Goal", "Saved", "Blocked", "Off_Target"]
        for c_key in cat_order:
            sub = all_shots_plot[all_shots_plot['cat_key'] == c_key]
            if sub.empty:
                continue
            first_row = sub.iloc[0]
            display_name = first_row['cat_name']
            col = first_row['cat_color']
            sym = first_row['cat_symbol']

            hover_txt = []
            sizes = []
            for _, r in sub.iterrows():
                min_val = int(r.get('minute', 0))
                p_name = _player_display_name(str(r.get('player_name', 'Cầu thủ')))
                t_name = str(r.get('team_name', ''))
                xg_val = float(r.get('shot_statsbomb_xg', 0.08))
                outcome_raw = str(r.get('shot_outcome_name', ''))
                dist = np.hypot(120.0 - r['location_x'], 40.0 - r['location_y']) * 0.9144

                sz = max(11, min(36, first_row['cat_base_sz'] + xg_val * 24))
                sizes.append(sz)
                hover_txt.append(
                    f"<b>{display_name}</b> ({outcome_raw})<br>"
                    f"👤 <b>{p_name}</b> ({t_name})<br>"
                    f"⏱️ Phút: <b>{min_val}'</b><br>"
                    f"🎯 xG: <b>{xg_val:.2f}</b><br>"
                    f"📍 Cự ly dứt điểm: <b>{dist:.1f} m</b>"
                )

            fig_shot.add_trace(go.Scatter(
                x=sub['plot_x'], y=sub['plot_y'], mode='markers',
                name=f"{display_name} ({len(sub)})",
                marker=dict(size=sizes, color=col, symbol=sym, line=dict(color='#0B0F19', width=1.6), opacity=0.94),
                text=hover_txt, hovertemplate="%{text}<extra></extra>"
            ))

    fig_shot.update_layout(
        xaxis=dict(range=[-5, 125], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        height=490, margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor="#111813", paper_bgcolor="#0E121A",
        font=dict(color="#E2E8F0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            bgcolor="rgba(19, 26, 38, 0.92)", bordercolor="#26354D", borderwidth=1,
            font=dict(size=12, color="#F8FAFC")
        )
    )

    col_map_view, col_xg_race = st.columns([1.35, 1.0], gap="medium")
    with col_map_view:
        with st.container(border=True):
            st.markdown("##### 🎯 Bản Đồ Tọa Độ Dứt Điểm 2D (Kích thước điểm = xG)")
            st.plotly_chart(fig_shot, width="stretch")

    with col_xg_race:
        with st.container(border=True):
            st.markdown("##### 📈 Biểu Đồ Đường Đua xG Theo Phút (FotMob xG Momentum)")
            fig_race = go.Figure()
            max_min_shot = int(max(90, pd.to_numeric(shots['minute'], errors='coerce').fillna(90).max()))
            max_cum_xg = 1.0
            for t_name, t_col, fill_rgba, icon_prefix in [
                (home_team, "#38BDF8", "rgba(56, 189, 248, 0.14)", "🔵"),
                (away_team, "#FF4D6D", "rgba(255, 77, 109, 0.14)", "🔴")
            ]:
                t_shots = shots[shots['team_name'] == t_name].copy()
                t_shots['minute_num'] = pd.to_numeric(t_shots['minute'], errors='coerce').fillna(0).astype(int)
                t_shots = t_shots.sort_values('minute_num')
                xs = [0] + t_shots['minute_num'].tolist() + [max_min_shot]
                cum_xg = t_shots['shot_statsbomb_xg'].cumsum().tolist()
                ys = [0.0] + [round(v, 2) for v in cum_xg] + ([round(cum_xg[-1], 2)] if cum_xg else [0.0])
                if ys:
                    max_cum_xg = max(max_cum_xg, max(ys))
                fig_race.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    name=f"{icon_prefix} {t_name} ({ys[-1]:.2f} xG)",
                    line=dict(color=t_col, width=3, shape="hv"),
                    fill="tozeroy", fillcolor=fill_rgba,
                    hovertemplate=f"<b>{t_name}</b><br>Phút %{{x}}'<br>xG tích lũy: <b>%{{y:.2f}}</b><extra></extra>"
                ))
                # Đánh dấu các bàn thắng trên đường đua xG
                if not t_shots.empty:
                    t_Shots_cum = t_shots.copy()
                    t_Shots_cum['cum_xg'] = t_Shots_cum['shot_statsbomb_xg'].cumsum()
                    goal_rows = t_Shots_cum[t_Shots_cum['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold().eq('goal')]
                    if not goal_rows.empty:
                        g_texts = [f"⚽ {_player_display_name(r.get('player_name', ''), pitch=True)} ({int(r['minute_num'])}')" for _, r in goal_rows.iterrows()]
                        t_pos = ["top left" if idx_g % 2 == 0 else "bottom right" for idx_g in range(len(goal_rows))]
                        fig_race.add_trace(go.Scatter(
                            x=goal_rows['minute_num'], y=goal_rows['cum_xg'],
                            mode="markers+text", text=g_texts, textposition=t_pos,
                            textfont=dict(size=10.5, color="#FFFFFF"),
                            marker=dict(size=13, color="#00E676", symbol="star", line=dict(color="#0B0F19", width=1.5)),
                            showlegend=False,
                            hovertemplate="<b>%{text}</b><br>xG tích lũy: %{y:.2f}<extra></extra>"
                        ))
            fig_race.add_vline(x=45, line_dash="dash", line_color="rgba(148, 163, 184, 0.35)", annotation_text="HT 45'", annotation_position="top")
            if max_min_shot > 95:
                fig_race.add_vline(x=90, line_dash="dash", line_color="rgba(250, 204, 21, 0.40)", annotation_text="FT 90'", annotation_position="top")
            fig_race.update_layout(
                xaxis=dict(title="Phút thi đấu", range=[0, max_min_shot + 4], gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False),
                yaxis=dict(title="xG tích lũy", range=[0, round(max_cum_xg * 1.24, 2)], gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False),
                height=490, margin=dict(l=10, r=10, t=36, b=10),
                plot_bgcolor="#0E131F", paper_bgcolor="#0E121A",
                font=dict(color="#E2E8F0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(19, 26, 38, 0.9)")
            )
            st.plotly_chart(fig_race, width="stretch")

    # Dòng thời gian chi tiết từng cú sút chuẩn FotMob / SofaScore (thay cho bảng xám đơn điệu)
    if not filtered_shots.empty:
        st.markdown("##### 📋 Dòng Thời Gian Dứt Điểm (FotMob Shot Log)")
        sorted_shots = filtered_shots.sort_values(by="minute", ascending=True)
        shot_cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; max-height: 440px; overflow-y: auto; padding-right: 4px;">'
        for _, sr in sorted_shots.iterrows():
            loc_x = float(sr['location_x'])
            loc_y = float(sr['location_y'])
            dist_m = np.hypot(120.0 - loc_x, 40.0 - loc_y) * 0.9144
            t_name = str(sr.get('team_name', ''))
            is_h = (t_name == home_team)
            p_raw = str(sr.get('player_name', 'Cầu thủ'))
            p_disp = html.escape(_player_display_name(p_raw))
            p_photo = get_player_avatar(p_raw, is_h, t_name)
            out_raw = str(sr.get('shot_outcome_name', 'Shot'))
            out_low = out_raw.strip().casefold()
            if out_low == 'goal':
                b_col, b_txt = "#00E676", "⚽ BÀN THẮNG"
            elif out_low in {'saved', 'saved to post'}:
                b_col, b_txt = "#38BDF8", "🧤 CỨU THUA"
            elif out_low == 'blocked':
                b_col, b_txt = "#FBBF24", "🛡️ BỊ CHẶN"
            else:
                b_col, b_txt = "#FF4D6D", "❌ RA NGOÀI"
            xg_v = float(sr.get('shot_statsbomb_xg', 0.0))
            min_v = int(pd.to_numeric(sr.get('minute', 0), errors='coerce') or 0)
            t_accent = "#38BDF8" if is_h else "#FF4D6D"
            shot_cards_html += (
                f'<div style="display: flex; align-items: center; gap: 11px; background: #131A26; border: 1px solid #243247; border-left: 3.5px solid {t_accent}; border-radius: 10px; padding: 9px 12px;">'
                f'<span style="font-weight: 900; color: #94A3B8; font-size: 13px; min-width: 32px;">{min_v}\'</span>'
                f'<img src="{p_photo}" width="38" height="38" style="border-radius: 50%; object-fit: cover; border: 1.5px solid {t_accent}; flex-shrink: 0;">'
                f'<div style="flex: 1; min-width: 0;">'
                f'<div style="color: #F8FAFC; font-weight: 800; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p_disp}</div>'
                f'<div style="color: #94A3B8; font-size: 11.5px;">{html.escape(t_name)} • {dist_m:.1f}m</div>'
                f'</div>'
                f'<div style="text-align: right; flex-shrink: 0;">'
                f'<div style="background: rgba(15, 23, 42, 0.9); border: 1px solid {b_col}; color: {b_col}; font-size: 10.5px; font-weight: 800; padding: 2px 7px; border-radius: 999px;">{b_txt}</div>'
                f'<div style="color: #E2E8F0; font-size: 12px; font-weight: 800; margin-top: 3px;">xG: <span style="color: #38BDF8;">{xg_v:.2f}</span></div>'
                f'</div>'
                f'</div>'
            )
        shot_cards_html += '</div>'
        st.markdown(shot_cards_html, unsafe_allow_html=True)


# ==========================================================
# 🏆 DANH MỤC GIẢI ĐẤU & TRẬN ĐẤU (SOFASCORE COMPETITION CATALOG)
# ==========================================================

_WC2018_SHOT_PLAYER_ALIASES = {
    "Raheem Sterling": "Raheem Shaquille Sterling",
    "Emil Peter Forsberg": "Emil Forsberg",
    "José Paolo Guerrero Gonzáles": "José Paolo Guerrero González",
    "Leon Aderemi Balogun": "Leon-Aderemi Balogun",
    "Fernando Luiz Roza": "Fernando Luiz Rosa",
    "Filipe Luís Kasmirski": "Filipe Luis Kasmirski",
    "Dylan Daniel Mahmoud Bronn": "Dylan Bronn",
    "Mathias Jørgensen Jatta": "Mathias Jattah-Njie Jørgensen",
}

@st.cache_data(show_spinner=False)
def load_wc2018_shots():
    shots_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worldcup_2018_shots.csv")
    if not os.path.exists(shots_path):
        return pd.DataFrame()
    try:
        df_s = pd.read_csv(shots_path, low_memory=False)
        if 'player_name' in df_s.columns:
            df_s['player_name'] = (
                df_s['player_name'].astype('string')
                .str.replace('"', "'", regex=False)
                .replace(_WC2018_SHOT_PLAYER_ALIASES)
            )
        return df_s
    except Exception:
        return pd.DataFrame()

wc2018_shots_df = load_wc2018_shots()

def _normalize_match_date(value):
    """Return a stable YYYY-MM-DD date, or None when the source has no date."""
    if value is None or pd.isna(value):
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    candidates = [raw_value[:10], raw_value.split()[0], raw_value]
    date_formats = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y")
    for candidate in dict.fromkeys(candidates):
        for date_format in date_formats:
            parsed = pd.to_datetime(candidate, format=date_format, errors="coerce")
            if not pd.isna(parsed):
                return parsed.strftime("%Y-%m-%d")
    return None

def _match_date_from_file(path):
    """Read a date field from an uploaded CSV when one is available."""
    if not path or not str(path).lower().endswith(".csv"):
        return None
    try:
        header_and_first_row = pd.read_csv(path, nrows=1, low_memory=False)
        date_columns = {str(column).strip().casefold(): column for column in header_and_first_row.columns}
        for candidate in ("match_date", "date", "fixture_date", "full_fixture_date"):
            column = date_columns.get(candidate)
            if column is not None and not header_and_first_row.empty:
                parsed = _normalize_match_date(header_and_first_row.iloc[0][column])
                if parsed:
                    return parsed
    except (OSError, ValueError, pd.errors.ParserError):
        pass
    return None

_WC2018_OWN_GOALS = {
    7529: [{"player": "Oghenekaro Etebo", "minute": 32, "team": "Croatia", "is_og": True}],
    7530: [{"player": "Aziz Behich", "minute": 81, "team": "France", "is_og": True}],
    7539: [{"player": "Thiago Cionek", "minute": 37, "team": "Senegal", "is_og": True}],
    7540: [{"player": "Ahmed Fathi", "minute": 47, "team": "Russia", "is_og": True}],
    7558: [{"player": "Denis Cheryshev", "minute": 23, "team": "Uruguay", "is_og": True}],
    7566: [{"player": "Edson Álvarez", "minute": 74, "team": "Sweden", "is_og": True}],
    7568: [{"player": "Yann Sommer", "minute": 90, "team": "Costa Rica", "is_og": True}],
    7569: [{"player": "Yassine Meriah", "minute": 33, "team": "Panama", "is_og": True}],
    7577: [{"player": "Aziz Bouhaddouz", "minute": 90, "team": "Iran", "is_og": True}],
    7582: [{"player": "Sergei Ignashevich", "minute": 12, "team": "Spain", "is_og": True}],
    8650: [{"player": "Fernandinho", "minute": 13, "team": "Belgium", "is_og": True}],
    8658: [{"player": "Mario Mandžukić", "minute": 18, "team": "France", "is_og": True}],
}

@st.cache_data(show_spinner=False)
def load_wc2018_raw():
    events_p = os.path.join(DATA_DIR, "worldcup_0_Extract.csv")
    if not os.path.exists(events_p):
        return pd.DataFrame(), []
    try:
        df = pd.read_csv(events_p, low_memory=False)
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        for _name_col in ('player_name', 'pass_recipient_name'):
            if _name_col in df.columns:
                df[_name_col] = df[_name_col].astype('string').str.replace('"', "'", regex=False)
    except Exception:
        return pd.DataFrame(), []

    matches = []
    if 'match_id' in df.columns:
        for mid, grp in df.groupby('match_id'):
            first_row = grp.iloc[0]
            home_team = str(first_row.get('home_team_home_team_name', ''))
            away_team = str(first_row.get('away_team_away_team_name', ''))
            h_score = int(first_row.get('home_score', 0)) if pd.notnull(first_row.get('home_score')) else 0
            a_score = int(first_row.get('away_score', 0)) if pd.notnull(first_row.get('away_score')) else 0
            stage = str(first_row.get('competition_stage_name', 'Group Stage'))
            stadium = str(first_row.get('stadium_name', 'FIFA World Cup Stadium'))
            referee = str(first_row.get('referee_name', 'FIFA Official Referee'))
            match_date = _normalize_match_date(first_row.get('match_date'))

            goals_info = []
            pen_home = None
            pen_away = None
            if not wc2018_shots_df.empty:
                m_all_shots = wc2018_shots_df[wc2018_shots_df['match_id'] == mid]
                m_reg_goals = m_all_shots[
                    (m_all_shots['period'].isin([1, 2, 3, 4])) &
                    (m_all_shots['shot_outcome_name'] == 'Goal')
                ]
                for _, s_row in m_reg_goals.iterrows():
                    p_name = str(s_row.get('player_name', 'Player'))
                    p_min = int(s_row.get('minute', 0))
                    p_team = str(s_row.get('team_name', ''))
                    goals_info.append({"player": p_name, "minute": p_min, "team": p_team, "is_og": False})

                m_p5 = m_all_shots[m_all_shots['period'] == 5]
                if not m_p5.empty:
                    m_p5_goals = m_p5[m_p5['shot_outcome_name'] == 'Goal']
                    pen_home = int((m_p5_goals['team_name'] == home_team).sum())
                    pen_away = int((m_p5_goals['team_name'] == away_team).sum())
            elif 'shot_outcome_name' in grp.columns:
                shots = grp[grp['shot_outcome_name'] == 'Goal']
                for _, s_row in shots.iterrows():
                    p_name = str(s_row.get('player_name', 'Player'))
                    p_min = int(s_row.get('minute', 0))
                    p_team = str(s_row.get('team_name', ''))
                    goals_info.append({"player": p_name, "minute": p_min, "team": p_team, "is_og": False})

            if int(mid) in _WC2018_OWN_GOALS:
                goals_info.extend(_WC2018_OWN_GOALS[int(mid)])
            goals_info.sort(key=lambda g: int(g.get("minute", 0)))

            pen_suffix = f" (Pen {pen_home}-{pen_away})" if pen_home is not None else ""
            label = f"{home_team} {h_score} - {a_score} {away_team}{pen_suffix}"
            matches.append({
                "id": mid,
                "label": label,
                "home_team": home_team, "away_team": away_team,
                "home_score": h_score, "away_score": a_score,
                "pen_home": pen_home, "pen_away": pen_away,
                "stage": f"World Cup 2018 - {stage}",
                "date": match_date,
                "stadium": stadium, "referee": referee,
                "goals": goals_info,
                "source_type": "wc2018_hdfs"
            })
    return df, matches

wc2018_events_full, wc2018_matches = load_wc2018_raw()

def load_match_stats_csv():
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worldcup_match_stats.csv")
    if not os.path.exists(stats_path): return pd.DataFrame()
    try:
        stats = pd.read_csv(stats_path, low_memory=False)
        if "match_id" in stats.columns:
            stats["match_id"] = pd.to_numeric(stats["match_id"], errors="coerce").astype("Int64")
        return stats
    except Exception: return pd.DataFrame()

match_stats_df = load_match_stats_csv()

# Xây dựng danh mục giải đấu phong cách SofaScore (Popular Competitions)
def _catalog_dir_signature():
    entries = []
    try:
        for fn in sorted(os.listdir(SAMPLE_DIR)):
            if fn.endswith((".json", ".csv")):
                fp = os.path.join(SAMPLE_DIR, fn)
                if os.path.isfile(fp):
                    entries.append((fn, int(os.path.getmtime(fp))))
        ucl_p = os.path.join(SAMPLE_DIR, "opta_ucl", "ucl_catalog.json")
        if os.path.isfile(ucl_p):
            entries.append(("opta_ucl/ucl_catalog.json", int(os.path.getmtime(ucl_p))))
    except OSError:
        pass
    return tuple(entries)

@st.cache_data(show_spinner=False)
def _build_sofascore_catalog_cached(_sig):
    catalog = {
        "🏆 FIFA World Cup": [
            {
                "id": "wc2022_final_arg_fra",
                "label": "Argentina 3 - 3 Pháp (Pen 4-2) ★",
                "home_team": "Argentina", "away_team": "France",
                "home_score": 3, "away_score": 3,
                "pen_home": 4, "pen_away": 2,
                "stage": "World Cup 2022 - Chung Kết",
                "date": "2022-12-18",
                "stadium": "Lusail Iconic Stadium (Qatar)",
                "referee": "Szymon Marciniak",
                "source_type": "json_file",
                "statsbomb_match_id": "3869685",
                "path": os.path.join(SAMPLE_DIR, "wc2022_final_arg_fra.json")
            },
            {
                "id": "wc2022_semifinal_arg_cro",
                "label": "Argentina 3 - 0 Croatia",
                "home_team": "Argentina", "away_team": "Croatia",
                "home_score": 3, "away_score": 0,
                "stage": "World Cup 2022 - Bán Kết",
                "date": "2022-12-13",
                "stadium": "Lusail Iconic Stadium (Qatar)",
                "referee": "Daniele Orsato",
                "source_type": "json_file",
                "statsbomb_match_id": "3869519",
                "path": os.path.join(SAMPLE_DIR, "wc2022_semifinal_arg_cro.json")
            }
        ] + [dict(m) for m in wc2018_matches],

        "⚽ UEFA Champions League": [
            {
                "id": "ucl2019_final_tot_liv",
                "label": "Tottenham 0 - 2 Liverpool ★",
                "home_team": "Tottenham Hotspur", "away_team": "Liverpool",
                "home_score": 0, "away_score": 2,
                "stage": "UEFA Champions League 2018/19 - Chung Kết",
                "date": "2019-06-01",
                "stadium": "Estadio Metropolitano (Madrid)",
                "referee": "Damir Skomina",
                "source_type": "json_file",
                "statsbomb_match_id": "22912",
                "path": os.path.join(SAMPLE_DIR, "ucl2019_final_tot_liv.json")
            }
        ],

        "🏆 UEFA European Championship": [
            {
                "id": "euro2024_final_esp_eng",
                "label": "Tây Ban Nha 2 - 1 Anh ★",
                "home_team": "Spain", "away_team": "England",
                "home_score": 2, "away_score": 1,
                "stage": "Euro 2024 - Chung Kết",
                "date": "2024-07-14",
                "stadium": "Olympiastadion Berlin (Đức)",
                "referee": "François Letexier",
                "source_type": "json_file",
                "statsbomb_match_id": "3943043",
                "path": os.path.join(SAMPLE_DIR, "euro2024_final_esp_eng.json")
            }
        ],

        "🏆 CONMEBOL Copa América": [],
        "🌍 Africa Cup of Nations (AFCON)": [],
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 English Premier League": [],

        "🇪🇸 Spanish La Liga": [
            {
                "id": "laliga2020_bar_eib",
                "label": "Barcelona 5 - 0 Eibar (Messi Poker) ★",
                "home_team": "Barcelona", "away_team": "Eibar",
                "home_score": 5, "away_score": 0,
                "stage": "La Liga 2019/20 - Vòng 25",
                "date": "2020-02-22",
                "stadium": "Camp Nou (Barcelona)",
                "referee": "César Soto Grado",
                "source_type": "json_file",
                "statsbomb_match_id": "303731",
                "path": os.path.join(SAMPLE_DIR, "laliga2020_bar_eib.json")
            }
        ],

        "🇩🇪 German Bundesliga": [],
        "🇫🇷 French Ligue 1": [],
        "🇮🇹 Italian Serie A": [],
        "📁 Trận đấu mới nạp (Custom Ingested)": []
    }

    # 1. Nạp danh mục mở rộng từ kho StatsBomb Open Data (586 trận: WC 2022 trọn bộ, Euro 2024/2020, Copa, AFCON, UCL Finals, Top 5 Leagues)
    manifest_path = os.path.join(SAMPLE_DIR, "statsbomb_catalog.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as _mf:
                _manifest = json.load(_mf)
            existing_sb_ids = {
                str(m.get("statsbomb_match_id"))
                for m_list in catalog.values()
                for m in m_list
                if m.get("statsbomb_match_id")
            }
            for item in (_manifest.get("matches") or []):
                sb_id = str(item.get("statsbomb_match_id") or "")
                if sb_id and sb_id in existing_sb_ids:
                    continue
                ev_full = os.path.join(SAMPLE_DIR, item.get("rel_path", ""))
                if not os.path.isfile(ev_full):
                    continue
                cat_name = item.get("category") or "📁 Trận đấu mới nạp (Custom Ingested)"
                rec = dict(item)
                rec["path"] = ev_full
                catalog.setdefault(cat_name, []).append(rec)
                if sb_id:
                    existing_sb_ids.add(sb_id)
        except Exception:
            pass

    # 2. Nạp kho dữ liệu chuyên sâu UEFA Champions League từ Opta Parquet (1.828 trận mùa 2015/16 – 2026/27)
    ucl_manifest_path = os.path.join(SAMPLE_DIR, "opta_ucl", "ucl_catalog.json")
    if os.path.isfile(ucl_manifest_path):
        try:
            with open(ucl_manifest_path, "r", encoding="utf-8") as _uf:
                _ucl_manifest = json.load(_uf)
            existing_ucl_date_scores = {
                (str(m.get("date") or "")[:10], tuple(sorted([int(m.get("home_score", 0) or 0), int(m.get("away_score", 0) or 0)])))
                for m in catalog["⚽ UEFA Champions League"]
            }
            for u_item in (_ucl_manifest.get("matches") or []):
                sig_ds = (
                    str(u_item.get("date") or "")[:10],
                    tuple(sorted([int(u_item.get("home_score", 0) or 0), int(u_item.get("away_score", 0) or 0)])),
                )
                if sig_ds in existing_ucl_date_scores and sig_ds[0] in ("2016-05-28", "2017-06-03", "2018-05-26", "2019-06-01"):
                    continue
                catalog["⚽ UEFA Champions League"].append(dict(u_item))
        except Exception:
            pass

    # 3. Tự động quét thêm các tệp tùy chỉnh người dùng kéo thả vào sample_data
    known_fnames = {
        "wc2022_final_arg_fra.json", "match_3869685.json",
        "wc2022_semifinal_arg_cro.json", "euro2024_final_esp_eng.json",
        "ucl2019_final_tot_liv.json", "laliga2020_bar_eib.json",
        "statsbomb_catalog.json"
    }
    for fname in os.listdir(SAMPLE_DIR):
        if fname in known_fnames or not (fname.endswith(".json") or fname.endswith(".csv")):
            continue
        fpath = os.path.join(SAMPLE_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        label = fname.replace('.json', '').replace('.csv', '')
        try:
            if fname.endswith(".json"):
                with open(fpath, "r", encoding="utf-8") as _cf:
                    _raw_c = json.load(_cf)
                if isinstance(_raw_c, list) and _raw_c:
                    _parsed_c = parse_statsbomb_events_json(_raw_c, fname)
                    label = f"{_parsed_c['home_team']} {_parsed_c['home_score']} - {_parsed_c['away_score']} {_parsed_c['away_team']}"
            elif fname.endswith(".csv"):
                _df_c = pd.read_csv(fpath, low_memory=False)
                _parsed_c = parse_uploaded_csv(_df_c)
                label = f"{_parsed_c['home_team']} {_parsed_c['home_score']} - {_parsed_c['away_score']} {_parsed_c['away_team']}"
        except Exception:
            pass
        catalog["📁 Trận đấu mới nạp (Custom Ingested)"].append({
            "id": fname,
            "label": label,
            "stage": "Custom Data",
            "source_type": "json_file" if fname.endswith(".json") else "csv_file",
            "path": fpath
        })

    _SB_TEAM_SHORT = {
        "Manchester City": "Man City",
        "Manchester United": "Man Utd",
        "Tottenham Hotspur": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Newcastle United": "Newcastle",
        "Brighton & Hove Albion": "Brighton",
        "Nottingham Forest": "Nott'm Forest",
        "West Bromwich Albion": "West Brom",
        "Sheffield United": "Sheffield Utd",
        "Charlton Athletic": "Charlton",
        "Blackburn Rovers": "Blackburn",
        "Bolton Wanderers": "Bolton",
        "Birmingham City": "Birmingham",
        "Leicester City": "Leicester",
        "Inter Milan": "Inter",
        "FC Internazionale Milano": "Inter",
        "Bayern Munich": "Bayern",
        "FC Bayern München": "Bayern",
        "Borussia Dortmund": "Dortmund",
        "Bayer Leverkusen": "Leverkusen",
        "Borussia Mönchengladbach": "M'gladbach",
        "Eintracht Frankfurt": "Frankfurt",
        "Paris Saint-Germain": "PSG",
        "Olympique de Marseille": "Marseille",
        "Atlético Madrid": "Atlético",
        "Deportivo Alavés": "Alavés",
        "Real Valladolid": "Valladolid",
        "Red Star Belgrade": "Crvena zvezda",
        "Shakhtar Donetsk": "Shakhtar",
        "Slovan Bratislava": "Slovan",
        "Lokomotiv Moscow": "Lokomotiv",
        "Sheriff Tiraspol": "Sheriff",
        "Hapoel Be'er Sheva": "H. Be'er Sheva",
        "Víkingur Reykjavík": "Víkingur",
        "United States": "USA",
        "Czech Republic": "Czechia",
        "Cape Verde Islands": "Cape Verde",
        "Equatorial Guinea": "Eq. Guinea",
        "North Macedonia": "N. Macedonia",
        "Bosnia and Herzegovina": "Bosnia",
        "Democratic Republic of Congo": "DR Congo",
        "Côte d'Ivoire": "Ivory Coast",
    }

    def _short_sb(t_name: str) -> str:
        s = str(t_name or "").strip()
        if s in _SB_TEAM_SHORT:
            return _SB_TEAM_SHORT[s]
        return s[:15].rstrip() if len(s) > 16 else s

    # Sắp xếp mới nhất lên đầu & đưa [Đội Nhà Tỷ-Số Đội Khách · DD/MM/YY] lên đầu để thấy ngay đối thủ!
    for matches in catalog.values():
        for match in matches:
            raw_d = str(match.get("date") or "").strip()
            if len(raw_d) == 10 and raw_d[4] == "-" and raw_d[7] == "-":
                match_date = raw_d
                short_date = f"{raw_d[8:10]}/{raw_d[5:7]}/{raw_d[2:4]}"
            else:
                match_date = _normalize_match_date(match.get("date")) or _match_date_from_file(match.get("path"))
                short_date = pd.to_datetime(match_date).strftime("%d/%m/%y") if match_date else None
            match["date"] = match_date

            ht_full = str(match.get("home_team") or "").strip()
            at_full = str(match.get("away_team") or "").strip()
            if ht_full and at_full and match.get("home_score") is not None and match.get("away_score") is not None:
                star_sfx = " ★" if "★" in str(match.get("label") or "") else ""
                pen_sfx = f" (P{match['pen_home']}-{match['pen_away']})" if match.get("pen_home") is not None else ""
                core_lbl = f"{_short_sb(ht_full)} {match['home_score']}-{match['away_score']} {_short_sb(at_full)}{pen_sfx}{star_sfx}"
            else:
                core_lbl = str(match.get("label") or "").strip()

            match["label"] = f"{core_lbl} · {short_date}" if short_date else core_lbl
        matches.sort(key=lambda match: (match.get("date") is not None, match.get("date") or ""), reverse=True)

    return catalog

def build_sofascore_catalog():
    return _build_sofascore_catalog_cached(_catalog_dir_signature())

catalog = build_sofascore_catalog()


# ==========================================================
# 📱 SIDEBAR: SOFASCORE POPULAR COMPETITIONS & FIXTURES
# ==========================================================

ui_lang, active_workspace = render_sidebar_pro_controls(catalog)

st.sidebar.markdown("""
<div class="sidebar-filter-title">🎛️ Bộ lọc trận đấu</div>
""", unsafe_allow_html=True)

# 1. Chọn giải đấu (Popular Competitions)
comp_names = [k for k in catalog.keys() if len(catalog[k]) > 0]
_total_catalog_matches = sum(len(catalog[k]) for k in comp_names)
st.sidebar.caption(f"📚 Kho Big Data hiện có: **{_total_catalog_matches:,} trận** • **{len(comp_names)} giải đấu**")

_jmp_comp = st.session_state.pop("jump_comp", None)
_jmp_match = st.session_state.pop("jump_match_label", None)
if _jmp_comp and _jmp_comp in comp_names:
    st.session_state["sidebar_comp_sel"] = _jmp_comp
if _jmp_match:
    st.session_state["sidebar_season_sel"] = "ALL"
if "sidebar_comp_sel" in st.session_state and st.session_state["sidebar_comp_sel"] not in comp_names:
    st.session_state["sidebar_comp_sel"] = comp_names[0]
selected_comp = st.sidebar.selectbox(
    "Chọn Giải Đấu:",
    comp_names,
    index=0,
    format_func=lambda k: f"{k} ({len(catalog[k]):,} trận)",
    key="sidebar_comp_sel",
)

if _jmp_match:
    st.session_state["sidebar_team_sel"] = "ALL"

all_comp_matches = catalog[selected_comp]
_season_counts = {}
for _m_item in all_comp_matches:
    _s_prefix = str(_m_item.get("stage") or "Khác").split(" - ")[0].strip() or "Khác"
    _season_counts[_s_prefix] = _season_counts.get(_s_prefix, 0) + 1

def _clean_season_disp(sk: str) -> str:
    if sk == "ALL":
        return f"Tất cả mùa giải ({len(all_comp_matches):,} trận)"
    short_sk = sk.replace("UEFA Champions League ", "Mùa ").replace("Champions League ", "Chung Kết ")
    return f"{short_sk} ({_season_counts[sk]:,} trận)"

if len(_season_counts) > 1:
    _season_keys = ["ALL"] + list(_season_counts.keys())
    if "sidebar_season_sel" in st.session_state and st.session_state["sidebar_season_sel"] not in _season_keys:
        st.session_state["sidebar_season_sel"] = "ALL"
    selected_season_key = st.sidebar.selectbox(
        "🗓️ Mùa giải / Năm:",
        _season_keys,
        index=0,
        format_func=_clean_season_disp,
        key="sidebar_season_sel",
    )
    season_matches = (
        all_comp_matches
        if selected_season_key == "ALL"
        else [m for m in all_comp_matches if (str(m.get("stage") or "Khác").split(" - ")[0].strip() or "Khác") == selected_season_key]
    )
else:
    season_matches = all_comp_matches

# 2. Lọc nhanh theo Đội bóng (giúp tìm trận trong 1 giây không cần cuộn dài)
_team_counts = {}
for _m_t in season_matches:
    for _tk in ("home_team", "away_team"):
        _tn = str(_m_t.get(_tk) or "").strip()
        if _tn:
            _team_counts[_tn] = _team_counts.get(_tn, 0) + 1

if len(_team_counts) > 2:
    _team_keys = ["ALL"] + sorted(_team_counts.keys())
    if "sidebar_team_sel" in st.session_state and st.session_state["sidebar_team_sel"] not in _team_keys:
        st.session_state["sidebar_team_sel"] = "ALL"
    selected_team_key = st.sidebar.selectbox(
        "🛡️ Lọc theo Đội bóng (gõ tên đội):",
        _team_keys,
        index=0,
        format_func=lambda tk: f"Tất cả các đội ({len(season_matches):,} trận)" if tk == "ALL" else f"{tk} ({_team_counts[tk]} trận)",
        key="sidebar_team_sel",
    )
    comp_matches = (
        season_matches
        if selected_team_key == "ALL"
        else [m for m in season_matches if m.get("home_team") == selected_team_key or m.get("away_team") == selected_team_key]
    )
else:
    comp_matches = season_matches

match_labels = [m["label"] for m in comp_matches]

# 3. Chọn trận đấu trong giải đấu đó (SofaScore Match Fixtures)
if match_labels:
    if _jmp_match and _jmp_match in match_labels:
        st.session_state["sidebar_match_sel"] = _jmp_match
    if "sidebar_match_sel" in st.session_state and st.session_state["sidebar_match_sel"] not in match_labels:
        st.session_state["sidebar_match_sel"] = match_labels[0]
    selected_match_label = st.sidebar.selectbox("⚽ Trận đấu (gõ đối thủ để tìm):", match_labels, index=0, key="sidebar_match_sel")
    selected_match_meta = next(m for m in comp_matches if m["label"] == selected_match_label)
else:
    selected_match_meta = catalog["🏆 FIFA World Cup"][0]

# 3. Nạp thêm trận đấu mới (File cục bộ hoặc Tải trực tiếp từ GitHub StatsBomb)
with st.sidebar.expander("➕ Nhập / Tải thêm trận đấu mới", expanded=False):
    st.caption("Kéo thả tệp `.json` / `.csv` hoặc nhập mã `Match ID` từ kho StatsBomb Open Data để tải tự động!")
    uploaded = st.file_uploader("Kéo thả tệp dữ liệu:", type=["json", "csv"], key="sidebar_file_uploader")
    if uploaded is not None:
        save_p = os.path.join(SAMPLE_DIR, uploaded.name)
        with open(save_p, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Đã lưu {uploaded.name}! Trận đấu đã được thêm vào danh mục.")
        st.rerun()
    st.markdown("---")
    sb_mid_in = st.text_input("🌐 Tải nhanh theo StatsBomb Match ID:", placeholder="VD: 3869684, 3942819, 3788741...", key="sb_direct_id")
    if st.button("⬇️ Tải tự động từ GitHub StatsBomb", width="stretch", key="btn_sb_pull"):
        mid_clean = str(sb_mid_in or "").strip()
        if not mid_clean.isdigit():
            st.warning("Vui lòng nhập mã số Match ID hợp lệ (chỉ gồm chữ số).")
        else:
            try:
                ev_dir = os.path.join(SAMPLE_DIR, "events")
                lu_dir = os.path.join(SAMPLE_DIR, "lineups")
                os.makedirs(ev_dir, exist_ok=True)
                os.makedirs(lu_dir, exist_ok=True)
                ev_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid_clean}.json"
                lu_url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/lineups/{mid_clean}.json"
                with urllib.request.urlopen(ev_url, timeout=25) as r_ev:
                    raw_ev_bytes = r_ev.read()
                raw_ev_json = json.loads(raw_ev_bytes.decode("utf-8"))
                save_ev_p = os.path.join(SAMPLE_DIR, f"match_{mid_clean}.json")
                with open(save_ev_p, "wb") as f_out:
                    f_out.write(raw_ev_bytes)
                try:
                    with urllib.request.urlopen(lu_url, timeout=20) as r_lu:
                        with open(os.path.join(lu_dir, f"{mid_clean}.json"), "wb") as f_lu:
                            f_lu.write(r_lu.read())
                except Exception:
                    pass
                parsed_new = parse_statsbomb_events_json(raw_ev_json, mid_clean)
                st.success(f"✅ Đã tải thành công: {parsed_new['home_team']} {parsed_new['home_score']} - {parsed_new['away_score']} {parsed_new['away_team']}!")
                st.rerun()
            except Exception as dl_err:
                st.error(f"Không tìm thấy Match ID {mid_clean} trên kho mở StatsBomb ({dl_err}).")


# ==========================================================
# ⚡ NẠP VÀ PHÂN TÍCH DỮ LIỆU CỦA TRẬN ĐẤU ĐƯỢC CHỌN
# ==========================================================

current_match = {}
match_events = pd.DataFrame()
custom_stats_dict = None
custom_lineups = {}
match_squads = {}

src_type = selected_match_meta.get("source_type")

if src_type == "json_file":
    fpath = selected_match_meta.get("path")
    with open(fpath, "r", encoding="utf-8") as f:
        raw_evs = json.load(f)
    parsed = parse_statsbomb_events_json(raw_evs, selected_match_meta.get("id"))
    match_events = parsed['events_df']
    custom_stats_dict = parsed['stats_dict']
    custom_lineups = parsed['starting_lineups']
    current_match = {
        "id": selected_match_meta.get("id"),
        "stage": selected_match_meta.get("stage", "Chung Kết"),
        "home_team": parsed['home_team'],
        "away_team": parsed['away_team'],
        "home_score": parsed['home_score'],
        "away_score": parsed['away_score'],
        "pen_home": parsed.get('pen_home') if parsed.get('pen_home') is not None else selected_match_meta.get("pen_home"),
        "pen_away": parsed.get('pen_away') if parsed.get('pen_away') is not None else selected_match_meta.get("pen_away"),
        "stadium": selected_match_meta.get("stadium", "Sân vận động quốc tế"),
        "referee": selected_match_meta.get("referee", "Trọng tài quốc tế"),
        "goals": parsed['goals']
    }

elif src_type == "csv_file":
    fpath = selected_match_meta.get("path")
    df_raw = pd.read_csv(fpath, low_memory=False)
    parsed = parse_uploaded_csv(df_raw)
    match_events = parsed['events_df']
    custom_stats_dict = parsed['stats_dict']
    custom_lineups = parsed['starting_lineups']
    current_match = {
        "id": selected_match_meta.get("id"),
        "stage": selected_match_meta.get("stage", "Custom Match"),
        "home_team": parsed['home_team'],
        "away_team": parsed['away_team'],
        "home_score": parsed['home_score'],
        "away_score": parsed['away_score'],
        "pen_home": None,
        "pen_away": None,
        "stadium": selected_match_meta.get("stadium", "Sân vận động"),
        "referee": selected_match_meta.get("referee", "Trọng tài"),
        "goals": parsed['goals']
    }

elif src_type == "wc2018_hdfs":
    mid = selected_match_meta.get("id")
    match_events = wc2018_events_full[wc2018_events_full['match_id'] == mid].copy()
    if not wc2018_shots_df.empty:
        mid_shots = wc2018_shots_df[
            (wc2018_shots_df['match_id'] == mid) &
            (wc2018_shots_df['period'].isin([1, 2, 3, 4]))
        ].copy()
        if not mid_shots.empty:
            match_events = pd.concat([match_events, mid_shots], ignore_index=True)
    current_match = {
        "id": mid,
        "stage": selected_match_meta.get("stage", "FIFA World Cup 2018"),
        "home_team": selected_match_meta.get("home_team"),
        "away_team": selected_match_meta.get("away_team"),
        "home_score": selected_match_meta.get("home_score"),
        "away_score": selected_match_meta.get("away_score"),
        "pen_home": selected_match_meta.get("pen_home"),
        "pen_away": selected_match_meta.get("pen_away"),
        "stadium": selected_match_meta.get("stadium"),
        "referee": selected_match_meta.get("referee"),
        "goals": selected_match_meta.get("goals", [])
    }

elif src_type == "opta_ucl_parquet":
    match_events, custom_stats_dict, custom_lineups, match_squads, current_match = load_opta_ucl_match_bundle(
        selected_match_meta, SAMPLE_DIR
    )

# StatsBomb events only list the starting XI and substitutions. The companion
# lineup files add the full match squad, including unused substitutes.
lineup_match_id = selected_match_meta.get("statsbomb_match_id")
if not lineup_match_id:
    source_id = selected_match_meta.get("id")
    source_path = selected_match_meta.get("path", "")
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    possible_id = source_stem.rsplit("_", 1)[-1]
    if possible_id.isdigit():
        lineup_match_id = possible_id
    elif str(source_id).isdigit():
        lineup_match_id = str(source_id)

if lineup_match_id:
    lineup_path = os.path.join(SAMPLE_DIR, "lineups", f"{lineup_match_id}.json")
    if os.path.isfile(lineup_path):
        try:
            with open(lineup_path, "r", encoding="utf-8") as lineup_file:
                match_squads = parse_statsbomb_lineups_json(json.load(lineup_file))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            match_squads = {}

# --- TIỀN XỬ LÝ SỰ KIỆN TRẬN ĐẤU ---
match_event_types = set(match_events['type_name'].dropna().astype(str).unique()) if 'type_name' in match_events.columns else set()
match_has_only_passes = match_event_types == {"Pass"}

home_events = match_events[match_events['team_name'] == current_match['home_team']]
away_events = match_events[match_events['team_name'] == current_match['away_team']]

def _order_team_players_chronologically(team_events):
    """Sắp xếp cầu thủ theo thời điểm xuất hiện đầu tiên trong trận (ưu tiên 11 cầu thủ đá chính hiệp 1, thủ môn đứng đầu)."""
    if team_events.empty or 'player_name' not in team_events.columns:
        return []
    valid_df = team_events[team_events['player_name'].notna()].copy()
    valid_df['player_name'] = valid_df['player_name'].astype(str).str.strip()
    valid_df = valid_df[(valid_df['player_name'] != '') & (valid_df['player_name'].str.lower() != 'nan')]
    if valid_df.empty:
        return []

    valid_df['_p'] = pd.to_numeric(valid_df.get('period', 1), errors='coerce').fillna(1)
    valid_df['_m'] = pd.to_numeric(valid_df.get('minute', 0), errors='coerce').fillna(0)
    valid_df['_s'] = pd.to_numeric(valid_df.get('second', 0), errors='coerce').fillna(0)
    valid_df['_t'] = (valid_df['_p'] - 1) * 45.0 + valid_df['_m'] + valid_df['_s'] / 60.0

    player_meta = []
    for pname, grp in valid_df.groupby('player_name', sort=False):
        first_t = float(grp['_t'].min())
        first_p = int(grp['_p'].min())
        if 'position_name' in grp.columns:
            p1_grp = grp[grp['_p'] == 1]
            pos_vals = p1_grp['position_name'].dropna().astype(str) if not p1_grp.empty else grp['position_name'].dropna().astype(str)
            pos_vals = pos_vals[pos_vals.str.strip() != '']
            if pos_vals.empty:
                pos_vals = grp['position_name'].dropna().astype(str)
                pos_vals = pos_vals[pos_vals.str.strip() != '']
        else:
            pos_vals = pd.Series(dtype=str)
        mode_pos = pos_vals.mode().iloc[0] if not pos_vals.empty else ''
        is_gk = 'goalkeeper' in mode_pos.lower() or mode_pos == 'GK'
        player_meta.append((pname, first_p, first_t, is_gk, len(grp)))

    # Chọn chuẩn 1 Thủ môn xuất phát + 10 Cầu thủ tuyến trên xuất phát sớm nhất
    player_meta.sort(key=lambda item: (item[1], item[2], -item[4]))
    gks = [item for item in player_meta if item[3]]
    outfielders = [item for item in player_meta if not item[3]]
    if gks:
        starters = [gks[0]] + outfielders[:10]
        subs = gks[1:] + outfielders[10:]
    else:
        starters = outfielders[:11]
        subs = outfielders[11:]
    starters.sort(key=lambda item: (0 if item[3] else 1, item[1], item[2]))
    subs.sort(key=lambda item: (item[1], item[2]))
    return [item[0] for item in starters + subs]

if custom_lineups and current_match['home_team'] in custom_lineups:
    _home_starters = [p['name'] for p in custom_lineups[current_match['home_team']]['players']]
    _home_others = [p for p in _order_team_players_chronologically(home_events) if p not in _home_starters]
    home_players = _home_starters + _home_others
else:
    home_players = _order_team_players_chronologically(home_events)

if custom_lineups and current_match['away_team'] in custom_lineups:
    _away_starters = [p['name'] for p in custom_lineups[current_match['away_team']]['players']]
    _away_others = [p for p in _order_team_players_chronologically(away_events) if p not in _away_starters]
    away_players = _away_starters + _away_others
else:
    away_players = _order_team_players_chronologically(away_events)

_COMPLEMENTARY_SLOTS = {
    "Center Back": ["Right Center Back", "Left Center Back", "Center Back"],
    "Right Center Back": ["Right Center Back", "Left Center Back", "Center Back"],
    "Left Center Back": ["Left Center Back", "Right Center Back", "Center Back"],
    "Center Defensive Midfield": ["Right Defensive Midfield", "Left Defensive Midfield", "Center Defensive Midfield"],
    "Right Defensive Midfield": ["Right Defensive Midfield", "Left Defensive Midfield", "Center Defensive Midfield"],
    "Left Defensive Midfield": ["Left Defensive Midfield", "Right Defensive Midfield", "Center Defensive Midfield"],
    "Center Midfield": ["Right Center Midfield", "Left Center Midfield", "Center Midfield"],
    "Right Center Midfield": ["Right Center Midfield", "Left Center Midfield", "Center Midfield"],
    "Left Center Midfield": ["Left Center Midfield", "Right Center Midfield", "Center Midfield"],
    "Center Attacking Midfield": ["Center Attacking Midfield", "Right Attacking Midfield", "Left Attacking Midfield"],
    "Right Attacking Midfield": ["Right Attacking Midfield", "Left Attacking Midfield", "Center Attacking Midfield"],
    "Left Attacking Midfield": ["Left Attacking Midfield", "Right Attacking Midfield", "Center Attacking Midfield"],
    "Right Midfield": ["Right Midfield", "Right Wing", "Right Center Midfield"],
    "Left Midfield": ["Left Midfield", "Left Wing", "Left Center Midfield"],
    "Right Wing": ["Right Wing", "Right Midfield", "Right Attacking Midfield"],
    "Left Wing": ["Left Wing", "Left Midfield", "Left Attacking Midfield"],
    "Center Forward": ["Right Center Forward", "Left Center Forward", "Center Forward", "Second Striker"],
    "Right Center Forward": ["Right Center Forward", "Left Center Forward", "Center Forward"],
    "Left Center Forward": ["Left Center Forward", "Right Center Forward", "Center Forward"],
    "Secondary Striker": ["Second Striker", "Center Attacking Midfield", "Right Center Forward", "Left Center Forward"],
    "Second Striker": ["Second Striker", "Center Attacking Midfield", "Right Center Forward", "Left Center Forward"],
}

def _deconflict_pitch_coordinates(lineup, is_home):
    """Đảm bảo không có 2 cầu thủ nào cùng đội bị trùng hoặc đè lên nhau trên sân bóng 2D."""
    if len(lineup) <= 1:
        return lineup

    # Bước 1: Tách các vị trí bị trùng tên sang vị trí đối xứng còn trống
    pos_counts = {}
    for p in lineup:
        pos_counts[p["pos"]] = pos_counts.get(p["pos"], 0) + 1

    used_slots = set()
    for idx, p in enumerate(lineup):
        raw_pos = p["pos"]
        assigned_slot = raw_pos
        if pos_counts.get(raw_pos, 0) > 1 or raw_pos in used_slots:
            candidates = _COMPLEMENTARY_SLOTS.get(raw_pos, [raw_pos])
            for cand in candidates:
                if cand not in used_slots:
                    assigned_slot = cand
                    break
        used_slots.add(assigned_slot)
        x, y = get_tactical_coords(assigned_slot, idx, is_home)
        p["x"], p["y"] = float(x), float(y)

    # Bước 2: Chống đè tọa độ có tính đến tỷ lệ khung hình ngang/dọc của sân
    min_dist = 9.6
    x_min, x_max = (16.0, 56.5) if is_home else (63.5, 104.0)
    y_min, y_max = 14.0, 64.0

    for _ in range(30):
        moved = False
        for i in range(len(lineup)):
            for j in range(i + 1, len(lineup)):
                dx = lineup[j]["x"] - lineup[i]["x"]
                dy = lineup[j]["y"] - lineup[i]["y"]
                eff_dist = float(np.hypot(dx * 0.76, dy * 0.94))
                if eff_dist < min_dist:
                    moved = True
                    overlap = (min_dist - eff_dist) + 0.45
                    raw_dist = float(np.hypot(dx, dy))
                    if raw_dist < 0.01:
                        ux, uy = 0.0, 1.0
                    else:
                        ux = (dx / raw_dist) * 0.65
                        uy = (dy / raw_dist) * 1.05
                        norm = float(np.hypot(ux, uy)) or 1.0
                        ux, uy = ux / norm, uy / norm

                    i_gk = ("goalkeeper" in lineup[i]["pos"].lower() or lineup[i]["pos"] == "GK")
                    j_gk = ("goalkeeper" in lineup[j]["pos"].lower() or lineup[j]["pos"] == "GK")
                    if i_gk:
                        lineup[j]["x"] = float(np.clip(lineup[j]["x"] + ux * overlap, x_min, x_max))
                        lineup[j]["y"] = float(np.clip(lineup[j]["y"] + uy * overlap, y_min, y_max))
                    elif j_gk:
                        lineup[i]["x"] = float(np.clip(lineup[i]["x"] - ux * overlap, x_min, x_max))
                        lineup[i]["y"] = float(np.clip(lineup[i]["y"] - uy * overlap, y_min, y_max))
                    else:
                        lineup[i]["x"] = float(np.clip(lineup[i]["x"] - ux * overlap * 0.5, x_min, x_max))
                        lineup[i]["y"] = float(np.clip(lineup[i]["y"] - uy * overlap * 0.5, y_min, y_max))
                        lineup[j]["x"] = float(np.clip(lineup[j]["x"] + ux * overlap * 0.5, x_min, x_max))
                        lineup[j]["y"] = float(np.clip(lineup[j]["y"] + uy * overlap * 0.5, y_min, y_max))
        if not moved:
            break
    return lineup

def _get_team_stats(team_name):
    if custom_stats_dict and team_name in custom_stats_dict:
        return custom_stats_dict[team_name]
    if not match_stats_df.empty and {"match_id", "team_name"}.issubset(match_stats_df.columns):
        numeric_mid = pd.to_numeric(pd.Series([current_match['id']]), errors="coerce").iloc[0]
        rows = match_stats_df[
            match_stats_df["match_id"].eq(numeric_mid) &
            match_stats_df["team_name"].astype(str).eq(str(team_name))
        ]
        if not rows.empty:
            return rows.iloc[0]
    return None

home_source_stats = _get_team_stats(current_match['home_team'])
away_source_stats = _get_team_stats(current_match['away_team'])

def _safe_int_stat(stats_obj, col_name):
    if stats_obj is None:
        return None
    val = None
    if isinstance(stats_obj, dict):
        val = stats_obj.get(col_name)
    elif hasattr(stats_obj, 'index') and col_name in stats_obj.index:
        val = stats_obj[col_name]
    if val is None or pd.isna(val) or val == "":
        return None
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return None

h_team_gk_saves = _safe_int_stat(home_source_stats, "goalkeeper_saves")
a_team_gk_saves = _safe_int_stat(away_source_stats, "goalkeeper_saves")

def _infer_formation_from_lineup(lineup):
    """Suy luận sơ đồ chiến thuật thực tế từ vị trí của 10 cầu thủ tuyến trên khi nguồn không ghi sẵn formation."""
    if not lineup:
        return "4-3-3"
    def_pos = {
        "Right Back", "Left Back", "Right Wing Back", "Left Wing Back",
        "Center Back", "Right Center Back", "Left Center Back",
        "RB", "LB", "RWB", "LWB", "CB", "RCB", "LCB", "DF"
    }
    dm_pos = {
        "Center Defensive Midfield", "Right Defensive Midfield", "Left Defensive Midfield",
        "CDM", "RDM", "LDM", "DM"
    }
    cm_pos = {
        "Center Midfield", "Right Center Midfield", "Left Center Midfield",
        "Right Midfield", "Left Midfield", "CM", "RCM", "LCM", "RM", "LM", "MF"
    }
    amw_pos = {
        "Center Attacking Midfield", "Right Attacking Midfield", "Left Attacking Midfield",
        "Right Wing", "Left Wing", "Secondary Striker", "Second Striker",
        "CAM", "RAM", "LAM", "AM", "RW", "LW", "SS"
    }
    st_pos = {
        "Center Forward", "Right Center Forward", "Left Center Forward",
        "CF", "RCF", "LCF", "ST", "FW"
    }
    outfield = [
        str(p.get("pos", "")).strip() for p in lineup
        if p.get("pos") and "goalkeeper" not in str(p.get("pos", "")).lower() and p.get("pos") != "GK"
    ]
    if len(outfield) != 10:
        return "4-3-3"

    n_def = sum(1 for pos in outfield if pos in def_pos)
    n_dm = sum(1 for pos in outfield if pos in dm_pos)
    n_cm = sum(1 for pos in outfield if pos in cm_pos)
    n_amw = sum(1 for pos in outfield if pos in amw_pos)
    n_st = sum(1 for pos in outfield if pos in st_pos)
    n_cm += 10 - (n_def + n_dm + n_cm + n_amw + n_st)

    has_cam = any(pos in {"Center Attacking Midfield", "CAM", "AM", "Secondary Striker", "Second Striker", "SS"} for pos in outfield)
    n_wings = sum(1 for pos in outfield if pos in {"Right Wing", "Left Wing", "RW", "LW"})
    n_wb = sum(1 for pos in outfield if pos in {"Right Wing Back", "Left Wing Back", "RWB", "LWB"})

    if n_def == 4:
        if n_dm == 2 and (n_amw + n_cm) == 3 and n_st == 1:
            return "4-2-3-1"
        if (n_dm == 1 and n_cm + n_amw == 4 and n_st == 1) or (n_dm == 0 and n_cm == 1 and n_amw == 4 and n_st == 1):
            if n_wings == 2 and not has_cam:
                return "4-3-3"
            return "4-1-4-1"
        if (n_dm + n_cm) == 3 and (n_amw + n_st) == 3:
            if has_cam and n_st == 2:
                return "4-3-1-2"
            return "4-3-3"
        if (n_dm + n_cm) == 2 and n_amw == 3 and n_st == 1:
            return "4-2-3-1"
        if (n_dm + n_cm + n_amw) == 4 and n_st == 2:
            if n_dm == 2 and n_amw == 2:
                return "4-2-2-2"
            return "4-4-2"
        if (n_dm + n_cm + n_amw) == 5 and n_st == 1:
            return "4-5-1"
    elif n_def in (3, 5):
        if n_def == 5 and n_wb == 2:
            mid_total = 2 + n_dm + n_cm
            att_total = n_amw + n_st
            if mid_total == 4 and n_amw == 2 and n_st == 1:
                return "3-4-2-1"
            if mid_total == 4 and att_total == 3:
                return "3-4-3"
            if (mid_total + n_amw) == 5 and n_st == 2:
                return "3-5-2"
            if (mid_total + n_amw) == 6 and n_st == 1:
                return "5-4-1"
            return f"3-{mid_total}-{att_total}"
        mid_all = n_dm + n_cm + (n_amw if n_st >= 2 else 0)
        att_all = n_st + (0 if n_st >= 2 else n_amw)
        if mid_all > 0 and att_all > 0 and n_def + mid_all + att_all == 10:
            return f"{n_def}-{mid_all}-{att_all}"

    mid_g = n_dm + n_cm
    att_g = n_amw + n_st
    if att_g == 0:
        att_g, mid_g = 1, max(1, mid_g - 1)
    if mid_g == 0:
        mid_g, att_g = 1, max(1, att_g - 1)
    if n_def > 0 and n_def + mid_g + att_g == 10:
        return f"{n_def}-{mid_g}-{att_g}"
    return "4-3-3"

def _format_formation(raw_fmt, fallback_lineup=None):
    s = str(raw_fmt or "").strip()
    if s.isdigit() and 3 <= len(s) <= 5:
        return "-".join(list(s))
    if s:
        return s
    return _infer_formation_from_lineup(fallback_lineup)

def _build_single_player_entry(p, idx, team_events, is_home, team_name="", tactics_map=None, is_sub=False, team_gk_saves=None, team_goals_conceded=0):
    p_ev = team_events[team_events['player_name'] == p]
    t_player = (tactics_map or {}).get(p)
    pos_raw = t_player.get('position', '') if t_player else ''
    if not pos_raw and 'position_name' in p_ev.columns:
        p1_ev = p_ev[pd.to_numeric(p_ev.get('period', 1), errors='coerce').fillna(1) == 1]
        pos_vals = p1_ev['position_name'].dropna().astype(str) if not p1_ev.empty else p_ev['position_name'].dropna().astype(str)
        pos_vals = pos_vals[pos_vals.str.strip() != '']
        if not pos_vals.empty:
            pos_raw = str(pos_vals.mode().iloc[0])
        elif not p_ev['position_name'].dropna().empty:
            pos_raw = str(p_ev['position_name'].dropna().iloc[0])

    pos = pos_raw if pos_raw else ("GK" if idx == 0 else ("DF" if idx in [1, 2, 3, 4] else ("MF" if idx in [5, 6, 7] else "FW")))
    is_gk = "goalkeeper" in pos.lower() or pos == "GK"
    gk_saves_ovr = team_gk_saves if (is_gk and not is_sub) else None
    score, bd, goals, xg, passes, comp_passes, tackles = calculate_player_rating(
        p_ev, is_gk=is_gk, gk_saves_override=gk_saves_ovr, goals_conceded=(team_goals_conceded if is_gk else 0)
    )

    gk_saves_val = int(gk_saves_ovr or 0)
    gk_claims_val = 0
    if is_gk and 'type_name' in p_ev.columns:
        gk_ev = p_ev[p_ev['type_name'] == 'Goal Keeper']
        if not gk_ev.empty and 'gk_type_name' in gk_ev.columns:
            gkt = gk_ev['gk_type_name'].fillna('').astype(str).str.lower()
            ev_saves = int(gkt.str.contains('save', regex=False).sum())
            if gk_saves_ovr is None or ev_saves > gk_saves_val:
                gk_saves_val = ev_saves
            gk_claims_val = int(gkt.str.contains('claim|punch|collected|smother|keeper swept', regex=True).sum())

    player_shots = p_ev[p_ev['type_name'] == 'Shot'] if 'type_name' in p_ev.columns else pd.DataFrame()
    if not player_shots.empty and 'shot_outcome_name' in player_shots.columns:
        shot_outcomes = player_shots['shot_outcome_name'].fillna('').astype(str).str.strip().str.casefold()
        goals_count = int(shot_outcomes.isin({'goal', '1', 'true'}).sum())
    else:
        goals_count = int(goals or 0)

    player_passes = p_ev[p_ev['type_name'] == 'Pass'] if 'type_name' in p_ev.columns else pd.DataFrame()
    assists_count = (
        int(pd.to_numeric(player_passes['pass_goal_assist'], errors='coerce').fillna(0).eq(1).sum())
        if 'pass_goal_assist' in player_passes.columns else 0
    )
    yellow_cards = 0
    red_cards = 0
    for c_col in ('card_name', 'foul_committed_card_name', 'bad_behaviour_card_name'):
        if c_col in p_ev.columns:
            card_values = p_ev[c_col].fillna('').astype(str).str.strip().str.casefold()
            yellow_cards += int(card_values.str.contains('yellow', regex=False).sum())
            red_cards += int(card_values.str.contains('red|second yellow', regex=True).sum())

    xg_values = pd.to_numeric(p_ev['shot_statsbomb_xg'], errors="coerce").dropna() if 'shot_statsbomb_xg' in p_ev.columns else pd.Series(dtype=float)
    goals_display = goals if not player_shots.empty and 'shot_outcome_name' in p_ev.columns else (goals if goals > 0 else None)
    xg_display = round(float(xg_values.sum()), 2) if not xg_values.empty else None

    activity_count = len(p_ev[p_ev['type_name'] == 'Pass']) if match_has_only_passes and 'type_name' in p_ev.columns else len(p_ev)
    activity_label = "Đường chuyền" if match_has_only_passes else "Sự kiện"

    x, y = get_tactical_coords(pos_raw or pos, idx, is_home)
    photo = get_player_avatar(p, is_home, team_name)
    short_name = _player_display_name(p, pitch=True)
    display_name = _player_display_name(p)

    return {
        "name": p, "display_name": display_name, "short_name": short_name, "pos": pos,
        "rating": score, "x": x, "y": y,
        "photo": photo, "events_count": activity_count, "activity_label": activity_label, "breakdown": bd,
        "goals": goals_display, "xg": xg_display, "passes": f"{comp_passes}/{passes}", "tackles": tackles,
        "goals_count": goals_count, "assists_count": assists_count,
        "yellow_cards": yellow_cards, "red_cards": red_cards,
        "gk_saves": gk_saves_val, "gk_claims": gk_claims_val,
        "is_home": is_home, "team_name": team_name, "is_sub": is_sub
    }

def build_lineup_data(players, team_events, is_home, team_name="", tactics_info=None, team_gk_saves=None, team_goals_conceded=0):
    tactics_players = tactics_info.get('players', []) if tactics_info else []
    tactics_map = {p['name']: p for p in tactics_players}
    chosen_players = players[:11]
    lineup = [
        _build_single_player_entry(
            p, idx, team_events, is_home, team_name=team_name, tactics_map=tactics_map, is_sub=False,
            team_gk_saves=team_gk_saves, team_goals_conceded=team_goals_conceded
        )
        for idx, p in enumerate(chosen_players)
    ]
    return _deconflict_pitch_coordinates(lineup, is_home)

def build_subs_data(players, team_events, is_home, team_name="", team_gk_saves=None, team_goals_conceded=0):
    sub_players = players[11:]
    return [
        _build_single_player_entry(
            p, 11 + idx, team_events, is_home, team_name=team_name, tactics_map=None, is_sub=True,
            team_gk_saves=team_gk_saves, team_goals_conceded=team_goals_conceded
        )
        for idx, p in enumerate(sub_players)
    ]

home_tactics = custom_lineups.get(current_match['home_team'])
away_tactics = custom_lineups.get(current_match['away_team'])

home_lineup = build_lineup_data(
    home_players, home_events, is_home=True, team_name=current_match['home_team'],
    tactics_info=home_tactics, team_gk_saves=h_team_gk_saves, team_goals_conceded=int(current_match.get('away_score', 0) or 0)
)
away_lineup = build_lineup_data(
    away_players, away_events, is_home=False, team_name=current_match['away_team'],
    tactics_info=away_tactics, team_gk_saves=a_team_gk_saves, team_goals_conceded=int(current_match.get('home_score', 0) or 0)
)
h_formation_str = _format_formation((home_tactics or {}).get('formation'), fallback_lineup=home_lineup)
a_formation_str = _format_formation((away_tactics or {}).get('formation'), fallback_lineup=away_lineup)
home_subs_lineup = build_subs_data(
    home_players, home_events, is_home=True, team_name=current_match['home_team'],
    team_gk_saves=h_team_gk_saves, team_goals_conceded=int(current_match.get('away_score', 0) or 0)
)
away_subs_lineup = build_subs_data(
    away_players, away_events, is_home=False, team_name=current_match['away_team'],
    team_gk_saves=a_team_gk_saves, team_goals_conceded=int(current_match.get('home_score', 0) or 0)
)

all_lineup = home_lineup + away_lineup
all_selectable_players = home_lineup + away_lineup + home_subs_lineup + away_subs_lineup
max_rating = max([p['rating'] for p in all_selectable_players]) if all_selectable_players else 8.0


# ==========================================================
# 🏟️ SOFASCORE MATCH HEADER CARD
# ==========================================================

home_scorers = [g for g in current_match['goals'] if g.get('team') == current_match['home_team']]
away_scorers = [g for g in current_match['goals'] if g.get('team') == current_match['away_team']]

def _goal_scorer_rows(goals):
    if not goals:
        return '<div class="match-goal-empty">Chưa có bàn thắng</div>'

    rows = []
    for goal in goals:
        full_name = str(goal.get('player') or 'Chưa rõ cầu thủ').strip()
        display_name = html.escape(_player_display_name(full_name))
        full_name_attr = html.escape(full_name, quote=True)
        minute = html.escape(str(goal.get('minute', '?')))
        og_tag = ' <span style="color:#F97316;font-size:11px;font-weight:900;">(OG)</span>' if goal.get('is_og') else ''
        rows.append(
            '<div class="match-goal-row">'
            f'<span class="match-goal-minute">⚽ {minute}′</span>'
            f'<span class="match-goal-player" title="Tên đầy đủ: {full_name_attr}">{display_name}{og_tag}</span>'
            '</div>'
        )
    return ''.join(rows)

h_col = _get_team_colors(current_match['home_team'])[0]
a_col = _get_team_colors(current_match['away_team'])[0]
home_team_label = html.escape(str(current_match['home_team']))
away_team_label = html.escape(str(current_match['away_team']))
home_scorers_html = _goal_scorer_rows(home_scorers)
away_scorers_html = _goal_scorer_rows(away_scorers)
comp_clean = selected_comp.replace("🏆", "").strip()
stage_label = html.escape(f"{comp_clean} • {current_match['stage']}".upper())
stadium_label = html.escape(str(current_match['stadium']))
referee_label = html.escape(str(current_match['referee']))
selected_match_date = _normalize_match_date(selected_match_meta.get("date"))
match_date_label = pd.to_datetime(selected_match_date).strftime("%d/%m/%Y") if selected_match_date else "Chưa rõ ngày"

h_avg_rating = round(float(np.mean([p['rating'] for p in home_lineup])), 2) if home_lineup else 6.8
a_avg_rating = round(float(np.mean([p['rating'] for p in away_lineup])), 2) if away_lineup else 6.8
h_hdr_xg = _format_stat_val(home_source_stats, "xg", 2)
a_hdr_xg = _format_stat_val(away_source_stats, "xg", 2)
h_hdr_pos = _format_stat_val(home_source_stats, "possession_pct", 1, "%")
a_hdr_pos = _format_stat_val(away_source_stats, "possession_pct", 1, "%")
motm_player = next((p for p in all_selectable_players if p['rating'] == max_rating), None)
motm_html = (
    f' &nbsp;•&nbsp; 👑 Cầu thủ xuất sắc nhất (MOTM): <b style="color: #00E676;">{html.escape(motm_player["display_name"])} ({motm_player["rating"]}★)</b>'
    if motm_player else ""
)
pen_badge_html = (
    f'<div style="text-align: center; width: 100%;"><span class="pen-badge">PEN {current_match["pen_home"]} - {current_match["pen_away"]}</span></div>'
    if current_match.get("pen_home") is not None and current_match.get("pen_away") is not None else ""
)

if active_workspace == "ws_live":
    render_live_and_multimatch_workspace(catalog, current_match, match_events, wc2018_events_full, wc2018_shots_df)
    st.stop()
elif active_workspace == "ws_hub":
    render_tournament_hub_workspace(catalog, wc2018_events_full, wc2018_shots_df, _player_display_name)
    st.stop()
elif active_workspace == "ws_ai":
    render_compare_ai_and_architecture_workspace(
        all_selectable_players,
        match_events,
        current_match,
        match_stats_df,
        wc2018_events_full,
        wc2018_shots_df,
        render_fotmob_stat_bar,
    )
    st.stop()

st.markdown(_compact_html(f"""
<div class="match-header-card">
    <div class="match-top-meta">
        <div style="color: #38BDF8; font-weight: 800; font-size: 13.5px; letter-spacing: 0.8px; text-align: left;">
            🏆 {stage_label}
        </div>
        <div style="color: #E2E8F0; font-size: 13px; font-weight: 700; white-space: nowrap; text-align: center;">
            📅 {match_date_label}
        </div>
        <div style="color: #94A3B8; font-size: 13px; text-align: right;">
            🏟️ {stadium_label}
        </div>
    </div>
    <div class="match-score-row">
        <div class="match-team-col home">
            <div class="match-team-name home">
                <span>🔵</span><span>{home_team_label}</span>
            </div>
            <div class="team-meta-pills">
                <span class="team-meta-pill home-accent">xG: {h_hdr_xg}</span>
                <span class="team-meta-pill">Kiểm soát: {h_hdr_pos}</span>
                <span class="team-meta-pill">Sơ đồ: {h_formation_str}</span>
                <span class="team-meta-pill">ĐTB: {h_avg_rating:.2f}★</span>
            </div>
        </div>
        <div class="score-badge">
            <div class="score-numbers">
                <span class="score-digit home">{current_match['home_score']}</span>
                <span class="score-sep">-</span>
                <span class="score-digit away">{current_match['away_score']}</span>
            </div>
            {pen_badge_html}
        </div>
        <div class="match-team-col away">
            <div class="match-team-name away">
                <span>{away_team_label}</span><span>🔴</span>
            </div>
            <div class="team-meta-pills">
                <span class="team-meta-pill away-accent">xG: {a_hdr_xg}</span>
                <span class="team-meta-pill">Kiểm soát: {a_hdr_pos}</span>
                <span class="team-meta-pill">Sơ đồ: {a_formation_str}</span>
                <span class="team-meta-pill">ĐTB: {a_avg_rating:.2f}★</span>
            </div>
        </div>
    </div>
    <div class="match-goal-lists">
        <div class="match-goal-side home" aria-label="Bàn thắng đội {home_team_label}">
            <div class="match-goal-team">BÀN THẮNG · {home_team_label}</div>
            <div class="match-goal-rows">{home_scorers_html}</div>
        </div>
        <div class="match-goal-side away" aria-label="Bàn thắng đội {away_team_label}">
            <div class="match-goal-team">BÀN THẮNG · {away_team_label}</div>
            <div class="match-goal-rows">{away_scorers_html}</div>
        </div>
    </div>
    <div style="text-align: center; color: #94A3B8; font-size: 13px; padding-top: 4px;">
        👨‍⚖️ Trọng tài chính: <b style="color: #F8FAFC;">{referee_label}</b>{motm_html}
    </div>
</div>
"""), unsafe_allow_html=True)


render_match_action_bar(
    current_match=current_match,
    selected_comp=selected_comp,
    match_events=match_events,
    all_selectable_players=all_selectable_players,
    home_lineup=home_lineup,
    away_lineup=away_lineup,
    home_source_stats=home_source_stats,
    away_source_stats=away_source_stats,
    lang=ui_lang,
)

# ==========================================================
# 📊 CÁC TABS PHÂN TÍCH SOFASCORE CHUYÊN SÂU
# ==========================================================

tab_lineup, tab_stats, tab_shotmap, tab_player, tab_network, tab_ai_news = st.tabs([
    tr("tab_lineup", ui_lang),
    tr("tab_stats", ui_lang),
    tr("tab_shotmap", ui_lang),
    tr("tab_player", ui_lang),
    tr("tab_network", ui_lang),
    tr("tab_ai_news", ui_lang),
])

# ----------------------------------------------------------
# TAB 1: ĐỘI HÌNH RA SÂN & SƠ ĐỒ CHIẾN THUẬT
# ----------------------------------------------------------
with tab_lineup:
    st.subheader("📋 Sơ Đồ Chiến Thuật & Danh Sách Cầu Thủ Kèm Điểm Số")

    best_p = next((p for p in all_selectable_players if p['rating'] == max_rating), all_selectable_players[0] if all_selectable_players else None)
    match_sel_key = f"sel_player_{current_match['id']}"
    sb_key = f"sb_sel_player_{current_match['id']}"
    tab4_sel_key = f"player_analytics_select_{current_match['id']}"
    pitch_ver_key = f"pitch_ver_{current_match['id']}"
    if pitch_ver_key not in st.session_state:
        st.session_state[pitch_ver_key] = 0
    if match_sel_key not in st.session_state or not st.session_state[match_sel_key]:
        st.session_state[match_sel_key] = best_p['name'] if best_p else ""

    active_player_name = st.session_state[match_sel_key]
    active_player = next((p for p in all_selectable_players if p['name'] == active_player_name), best_p)
    if active_player:
        active_player_name = active_player['name']
        st.session_state[match_sel_key] = active_player_name
        st.session_state[sb_key] = active_player_name

    col_pitch, col_detail = st.columns([1.52, 1.48], gap="medium")

    with col_pitch:
        fig_pitch = go.Figure()
        # Thảm cỏ sọc sân vận động sắc nét chuẩn SofaScore Pro
        for i in range(12):
            stripe_col = "#121C15" if i % 2 == 0 else "#16241B"
            fig_pitch.add_shape(
                type="rect", x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80,
                fillcolor=stripe_col, line=dict(width=0), layer="below"
            )
        p_line = dict(color="rgba(255, 255, 255, 0.36)", width=2)
        fig_pitch.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=p_line, layer="below")
        fig_pitch.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=p_line, layer="below")
        fig_pitch.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=p_line, layer="below")
        fig_pitch.add_shape(type="circle", x0=59.3, y0=39.3, x1=60.7, y1=40.7, fillcolor="rgba(255, 255, 255, 0.55)", line=dict(width=0), layer="below")
        fig_pitch.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=p_line, layer="below")
        fig_pitch.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=p_line, layer="below")
        fig_pitch.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="rgba(255, 255, 255, 0.32)", width=1.6), layer="below")
        fig_pitch.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="rgba(255, 255, 255, 0.32)", width=1.6), layer="below")
        fig_pitch.add_shape(type="circle", x0=11.5, y0=39.5, x1=12.5, y1=40.5, fillcolor="rgba(255, 255, 255, 0.45)", line=dict(width=0), layer="below")
        fig_pitch.add_shape(type="circle", x0=107.5, y0=39.5, x1=108.5, y1=40.5, fillcolor="rgba(255, 255, 255, 0.45)", line=dict(width=0), layer="below")
        fig_pitch.add_shape(type="rect", x0=-2.2, y0=36, x1=0, y1=44, line=dict(color="#38BDF8", width=2), fillcolor="rgba(56,189,248,0.16)", layer="below")
        fig_pitch.add_shape(type="rect", x0=120, y0=36, x1=122.2, y1=44, line=dict(color="#FF4D6D", width=2), fillcolor="rgba(255,77,109,0.16)", layer="below")

        fig_pitch.add_annotation(
            x=28, y=84.2, text=f"🔵 {current_match['home_team'].upper()} ({h_formation_str}) • {h_avg_rating:.2f}★",
            showarrow=False, font=dict(size=11.5, color="#7DD3FC", family="Arial Black, sans-serif"),
            bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="rgba(56, 189, 248, 0.48)", borderwidth=1, borderpad=4
        )
        fig_pitch.add_annotation(
            x=92, y=84.2, text=f"{current_match['away_team'].upper()} ({a_formation_str}) • {a_avg_rating:.2f}★ 🔴",
            showarrow=False, font=dict(size=11.5, color="#FDA4AF", family="Arial Black, sans-serif"),
            bgcolor="rgba(15, 23, 42, 0.88)", bordercolor="rgba(255, 77, 109, 0.48)", borderwidth=1, borderpad=4
        )

        def _pitch_player_label(player, is_motm):
            event_markers = []
            event_markers.extend("⚽" for _ in range(min(player.get('goals_count', 0), 8)))
            for count, symbol in (
                (player.get('assists_count', 0), '👟'),
                (player.get('yellow_cards', 0), '🟨'),
                (player.get('red_cards', 0), '🟥'),
            ):
                if count:
                    event_markers.append(f"{symbol}{'×' + str(count) if count > 1 else ''}")
            marker_html = (
                f"<span style='font-size:10px; white-space:nowrap;'> {' '.join(event_markers)}</span>"
                if event_markers else ""
            )
            star = " ★" if is_motm else ""
            return (
                f"<b>{player['rating']}{star}</b>{marker_html}<br>"
                f"<span style='font-size:11px; color:#FFFFFF;'>{html.escape(player['short_name'])}</span>"
            )

        # Cầu thủ đội nhà (không hiện tooltip khi di chuột lướt qua, chỉ chọn khi bấm trực tiếp)
        for p in home_lineup:
            is_sel = (p['name'] == active_player_name)
            r_col = "#00D66C" if p['rating'] >= 7.5 else ("#FF9800" if p['rating'] >= 6.5 else "#F44336")
            border_col = "#00F0FF" if is_sel else r_col
            border_w = 4.2 if is_sel else 2.8
            fig_pitch.add_layout_image(
                source=p['photo'], xref="x", yref="y", x=p['x'], y=p['y'],
                sizex=6.8 if is_sel else 6.1, sizey=6.8 if is_sel else 6.1,
                xanchor="center", yanchor="middle",
                sizing="contain", layer="above"
            )
            fig_pitch.add_trace(go.Scatter(
                x=[p['x']], y=[p['y']], mode='markers+text',
                marker=dict(size=44 if is_sel else 40, color="rgba(0,0,0,0.01)", line=dict(color=border_col, width=border_w)),
                text=[_pitch_player_label(p, p['rating'] == max_rating)],
                textposition="top center",
                customdata=[p['name']],
                hoverinfo="none",
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0))
            ))

        # Cầu thủ đội khách (không hiện tooltip khi di chuột lướt qua, chỉ chọn khi bấm trực tiếp)
        for p in away_lineup:
            is_sel = (p['name'] == active_player_name)
            r_col = "#00D66C" if p['rating'] >= 7.5 else ("#FF9800" if p['rating'] >= 6.5 else "#F44336")
            border_col = "#00F0FF" if is_sel else r_col
            border_w = 4.2 if is_sel else 2.8
            fig_pitch.add_layout_image(
                source=p['photo'], xref="x", yref="y", x=p['x'], y=p['y'],
                sizex=6.8 if is_sel else 6.1, sizey=6.8 if is_sel else 6.1,
                xanchor="center", yanchor="middle",
                sizing="contain", layer="above"
            )
            fig_pitch.add_trace(go.Scatter(
                x=[p['x']], y=[p['y']], mode='markers+text',
                marker=dict(size=44 if is_sel else 40, color="rgba(0,0,0,0.01)", line=dict(color=border_col, width=border_w)),
                text=[_pitch_player_label(p, p['rating'] == max_rating)],
                textposition="top center",
                customdata=[p['name']],
                hoverinfo="none",
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0))
            ))

        fig_pitch.update_layout(
            xaxis=dict(range=[-4, 124], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
            yaxis=dict(range=[-5, 88], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
            height=515, margin=dict(l=4, r=4, t=8, b=6),
            plot_bgcolor="#131714", paper_bgcolor="#0D100E", showlegend=False,
            clickmode="event+select", dragmode=False
        )
        pitch_ev = st.plotly_chart(
            fig_pitch, width="stretch",
            on_select="rerun", selection_mode="points",
            key=f"pitch_chart_{current_match['id']}_{st.session_state[pitch_ver_key]}"
        )
        if pitch_ev:
            sel_obj = getattr(pitch_ev, "selection", None) or (pitch_ev.get("selection") if isinstance(pitch_ev, dict) else None)
            pts = (getattr(sel_obj, "points", None) or (sel_obj.get("points") if isinstance(sel_obj, dict) else [])) if sel_obj else []
            if pts:
                cdata = getattr(pts[0], "customdata", None) or (pts[0].get("customdata") if isinstance(pts[0], dict) else None)
                clicked_name = cdata[0] if (isinstance(cdata, list) and len(cdata) > 0) else (cdata if isinstance(cdata, str) else None)
                if clicked_name:
                    st.session_state[match_sel_key] = clicked_name
                    st.session_state[sb_key] = clicked_name
                    st.session_state[tab4_sel_key] = clicked_name
                    st.session_state[pitch_ver_key] = st.session_state.get(pitch_ver_key, 0) + 1
                    st.rerun()

        st.markdown(
            '<div style="display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px 12px; align-items:center;'
            ' margin:2px 0 8px; padding:7px 12px; border:1px solid #243247;'
            ' border-radius:9px; background:#121926; color:#CBD5E1; font-size:11.5px;">'
            '<span><b>⚽</b> Bàn thắng</span>'
            '<span><b>👟</b> Kiến tạo</span>'
            '<span><b>🟨</b> Thẻ vàng</span>'
            '<span><b>🟥</b> Thẻ đỏ</span>'
            '<span><b>🟦 Viền xanh</b>: Đang chọn</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Khối Top 3 cầu thủ nổi bật mỗi đội đặt ngay dưới sân để cân bằng chiều cao 2 cột
        top_home_3 = sorted(home_lineup + home_subs_lineup, key=lambda x: x['rating'], reverse=True)[:3]
        top_away_3 = sorted(away_lineup + away_subs_lineup, key=lambda x: x['rating'], reverse=True)[:3]

        def _top_performers_col_html(players_top, team_lbl, accent_c, dot_icon):
            rows_h = []
            for tp in players_top:
                rv = tp['rating']
                rc = "#00D66C" if rv >= 7.5 else ("#FF9800" if rv >= 6.5 else "#EF4444")
                ev_icons = ("⚽" * min(tp.get('goals_count', 0), 4)) + ("👟" * min(tp.get('assists_count', 0), 3))
                ev_span = f' <span style="font-size:11px;">{ev_icons}</span>' if ev_icons else ''
                rows_h.append(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;background:#0E1420;border:1px solid #1E293B;border-radius:8px;">'
                    f'<img src="{tp["photo"]}" width="30" height="30" style="border-radius:50%;object-fit:cover;border:1.5px solid {accent_c};flex:none;">'
                    f'<div style="min-width:0;flex:1;">'
                    f'<div style="color:#F8FAFC;font-weight:700;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(tp["display_name"])}{ev_span}</div>'
                    f'<div style="color:#64748B;font-size:10.5px;">{html.escape("Dự bị" if tp.get("is_sub") else str(tp["pos"]))}</div>'
                    f'</div>'
                    f'<span style="background:{rc};color:#000;font-weight:900;font-size:11.5px;padding:2px 6px;border-radius:5px;white-space:nowrap;">{rv}★</span>'
                    f'</div>'
                )
            return (
                f'<div style="min-width:0;">'
                f'<div style="color:{accent_c};font-size:11.5px;font-weight:800;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{dot_icon} TOP ĐIỂM · {html.escape(team_lbl.upper())}</div>'
                f'<div style="display:flex;flex-direction:column;gap:5px;">{"".join(rows_h)}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:#131A26;border:1px solid #243247;border-radius:12px;padding:10px 12px;">'
            f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;">'
            f'{_top_performers_col_html(top_home_3, current_match["home_team"], "#38BDF8", "🔵")}'
            f'{_top_performers_col_html(top_away_3, current_match["away_team"], "#FF4D6D", "🔴")}'
            f'</div></div>',
            unsafe_allow_html=True
        )

    with col_detail:
        player_names = [p['name'] for p in all_selectable_players]
        labels_map = {
            p['name']: f"{'🔄 ' if p.get('is_sub') else ''}{p['display_name']} ({p.get('team_name', '')} • {'Dự bị' if p.get('is_sub') else p['pos']} • {p['rating']}★)"
            for p in all_selectable_players
        }
        if st.session_state.get(sb_key) not in player_names and player_names:
            st.session_state[sb_key] = active_player_name if active_player_name in player_names else player_names[0]

        def _on_sb_player_change():
            st.session_state[match_sel_key] = st.session_state[sb_key]
            st.session_state[tab4_sel_key] = st.session_state[sb_key]
            st.session_state[pitch_ver_key] = st.session_state.get(pitch_ver_key, 0) + 1

        chosen_name = st.selectbox(
            "🔍 Chọn cầu thủ xem chi tiết (Đá chính & Dự bị):",
            player_names,
            format_func=lambda n: labels_map.get(n, n),
            key=sb_key,
            on_change=_on_sb_player_change
        )

        p_to_render = next((p for p in all_selectable_players if p['name'] == chosen_name), active_player)
        p_events_sel = match_events[match_events['player_name'] == p_to_render['name']] if 'player_name' in match_events.columns else pd.DataFrame()

        render_sofascore_player_panel(p_to_render, p_events_sel, current_match, max_rating=max_rating, match_events=match_events)

    st.markdown('<div style="margin: 14px 0 8px; border-top: 1px solid #1E293B;"></div>', unsafe_allow_html=True)
    st.markdown("#### 👥 Thay người & Danh sách dự bị")

    def _team_substitution_status(team_events, team_squad, ordered_players=None):
        """Combine the full roster with event substitutions to label on/off/unplayed players."""
        if 'type_name' not in team_events.columns:
            return [], [], [], False, False, False

        def _minute_label(value):
            try:
                return f"{int(float(value))}′" if pd.notna(value) and str(value).strip() else ""
            except (TypeError, ValueError):
                return ""

        def _clock_minute_label(value):
            if not value:
                return ""
            try:
                return f"{int(str(value).split(':', 1)[0])}′"
            except (TypeError, ValueError):
                return ""

        def _name_key(value):
            return _photo_name_key(str(value or '').strip())

        event_types = team_events['type_name'].fillna('').astype(str).str.strip().str.casefold()
        substitution_rows = team_events[event_types.eq('substitution')].copy()
        has_reason_data = 'substitution_outcome_name' in team_events.columns
        has_incoming_data = 'substitution_replacement_name' in team_events.columns
        has_roster = bool(team_squad and team_squad.get('players'))
        incoming_by_name = {}
        outgoing_by_name = {}
        injuries_by_name = {}

        if not substitution_rows.empty:
            minute_values = substitution_rows['minute'] if 'minute' in substitution_rows.columns else pd.Series(index=substitution_rows.index, dtype=float)
            substitution_rows['_minute_sort'] = pd.to_numeric(minute_values, errors='coerce')
            substitution_rows = substitution_rows.sort_values('_minute_sort', na_position='last')
            for _, event in substitution_rows.iterrows():
                incoming = str(event.get('substitution_replacement_name', '') or '').strip()
                outgoing = str(event.get('player_name', '') or '').strip()
                minute_label = _minute_label(event.get('minute'))
                outcome = str(event.get('substitution_outcome_name', '') or '').strip().casefold()

                if incoming and incoming.casefold() != 'nan':
                    incoming_by_name[_name_key(incoming)] = {
                        'name': incoming, 'minute': minute_label, 'counterpart': outgoing
                    }
                if outgoing and outgoing.casefold() != 'nan':
                    outgoing_by_name[_name_key(outgoing)] = {
                        'name': outgoing, 'minute': minute_label, 'counterpart': incoming
                    }
                    if outcome in {'injury', 'injured'}:
                        injuries_by_name[_name_key(outgoing)] = {
                            'name': outgoing, 'minute': minute_label, 'counterpart': incoming
                        }

        # Nếu dữ liệu không có dòng sự kiện Substitution (như World Cup 2018 HDFS),
        # suy luận chính xác cầu thủ vào sân và rời sân từ thời điểm sự kiện đầu/cuối.
        if not has_roster and not incoming_by_name and ordered_players and len(ordered_players) > 11:
            starters_list = ordered_players[:11]
            subs_list = ordered_players[11:]
            starter_exits = []
            for sp in starters_list:
                sp_ev = team_events[team_events['player_name'] == sp]
                sp_mins = pd.to_numeric(sp_ev.get('minute'), errors='coerce').dropna() if 'minute' in sp_ev.columns else pd.Series(dtype=float)
                if not sp_mins.empty:
                    max_m = int(sp_mins.max())
                    if max_m <= 86:
                        starter_exits.append((sp, max_m))
            starter_exits.sort(key=lambda x: x[1])
            starter_exits = starter_exits[:len(subs_list)]

            for idx_s, sub_name in enumerate(subs_list):
                sub_ev = team_events[team_events['player_name'] == sub_name]
                sub_mins = pd.to_numeric(sub_ev.get('minute'), errors='coerce').dropna() if 'minute' in sub_ev.columns else pd.Series(dtype=float)
                min_in = int(sub_mins.min()) if not sub_mins.empty else 60
                counterpart = starter_exits[idx_s][0] if idx_s < len(starter_exits) else ''
                sub_min_label = f"{min_in}′"
                incoming_by_name[_name_key(sub_name)] = {
                    'name': sub_name, 'minute': sub_min_label, 'counterpart': counterpart
                }
                if counterpart:
                    outgoing_by_name[_name_key(counterpart)] = {
                        'name': counterpart, 'minute': sub_min_label, 'counterpart': sub_name
                    }
            has_incoming_data = True

        roster_players = (team_squad or {}).get('players', [])
        bench = []
        outgoing_players = []
        for player in roster_players:
            if player.get('injured_on_exit'):
                name = player.get('name', '').strip()
                if name:
                    key = _name_key(name)
                    exit_event = outgoing_by_name.get(key, {})
                    injuries_by_name.setdefault(key, {
                        'name': name,
                        'minute': exit_event.get('minute') or _clock_minute_label(player.get('minute_out')),
                        'counterpart': exit_event.get('counterpart', ''),
                    })

        if has_roster:
            for player in roster_players:
                name = player.get('name', '').strip()
                if not name:
                    continue
                key = _name_key(name)
                if not player.get('started'):
                    entry = incoming_by_name.get(key)
                    entered = bool(entry or player.get('entered'))
                    minute = (entry or {}).get('minute') or _clock_minute_label(player.get('minute_in'))
                    replacement_for = _player_display_name((entry or {}).get('counterpart', '')) if (entry or {}).get('counterpart') else ''
                    bench.append({
                        'name': name,
                        'minute': minute if entered else '',
                        'status_label': '⬆ VÀO SÂN' if entered else 'DỰ BỊ',
                        'status_color': '#00D66C' if entered else '#94A3B8',
                        'detail': f"Thay {replacement_for}" if entered and replacement_for else ('Đã vào sân' if entered else 'Chưa vào sân'),
                        'sort_minute': int(minute[:-1]) if minute.endswith('′') and minute[:-1].isdigit() else 9999,
                    })

                exit_event = outgoing_by_name.get(key)
                left = bool(exit_event or player.get('left'))
                if left:
                    minute = (exit_event or {}).get('minute') or _clock_minute_label(player.get('minute_out'))
                    replacement = _player_display_name((exit_event or {}).get('counterpart', '')) if (exit_event or {}).get('counterpart') else ''
                    outgoing_players.append({
                        'name': name,
                        'minute': minute,
                        'status_label': '⬇ RỜI SÂN',
                        'status_color': '#FF6577',
                        'detail': f"Nhường {replacement}" if replacement else 'Rời sân',
                        'sort_minute': int(minute[:-1]) if minute.endswith('′') and minute[:-1].isdigit() else 9999,
                    })
        else:
            bench = [{
                'name': item['name'], 'minute': item['minute'],
                'status_label': '⬆ VÀO SÂN', 'status_color': '#00D66C',
                'detail': f"Thay {_player_display_name(item['counterpart'])}" if item['counterpart'] else 'Đã vào sân',
                'sort_minute': int(item['minute'][:-1]) if item['minute'].endswith('′') and item['minute'][:-1].isdigit() else 9999,
            } for item in incoming_by_name.values()]
            outgoing_players = [{
                'name': item['name'], 'minute': item['minute'],
                'status_label': '⬇ RỜI SÂN', 'status_color': '#FF6577',
                'detail': f"Nhường {_player_display_name(item['counterpart'])}" if item['counterpart'] else 'Rời sân',
                'sort_minute': int(item['minute'][:-1]) if item['minute'].endswith('′') and item['minute'][:-1].isdigit() else 9999,
            } for item in outgoing_by_name.values()]

        bench.sort(key=lambda item: (item['sort_minute'], item['name']))
        outgoing_players.sort(key=lambda item: (item['sort_minute'], item['name']))
        injuries = [{
            'name': item['name'], 'minute': item['minute'],
            'status_label': '🩹 CHẤN THƯƠNG', 'status_color': '#FFB547',
            'detail': f"Thay bởi {_player_display_name(item['counterpart'])}" if item['counterpart'] else 'Chấn thương',
        } for item in injuries_by_name.values()]
        injuries.sort(key=lambda item: (
            int(item['minute'][:-1]) if item['minute'].endswith('′') and item['minute'][:-1].isdigit() else 9999,
            item['name'],
        ))
        return bench, outgoing_players, injuries, has_roster, bool(substitution_rows.empty is False or has_incoming_data), has_reason_data

    _selectable_by_key = {_photo_name_key(p['name']): p for p in all_selectable_players}

    def _render_status_grid(items, is_home, team_name):
        accent = '#38BDF8' if is_home else '#FF4D6D'
        if not items:
            return
        cards = []
        for item in items:
            name = item['name']
            disp_n = _player_display_name(name)
            photo = get_player_avatar(name, is_home, team_name)
            p_entry = _selectable_by_key.get(_photo_name_key(name))
            rating_badge_html = ""
            if p_entry and p_entry.get('events_count', 0) > 0:
                rv = p_entry['rating']
                rc = "#00D66C" if rv >= 7.5 else ("#FF9800" if rv >= 6.5 else "#EF4444")
                rating_badge_html = f"<span style='background:{rc};color:#000;font-weight:900;font-size:11px;padding:2px 6px;border-radius:5px;white-space:nowrap;'>{rv}★</span>"
            minute_badge = (
                f"<span style='white-space:nowrap;background:#1E293B;border:1px solid #334155;color:#F8FAFC;border-radius:6px;padding:2px 6px;font-size:11px;font-weight:800;'>{html.escape(item['minute'])}</span>"
                if item.get('minute') else ''
            )
            cards.append(
                f'<div style="display:flex;align-items:center;gap:9px;background:#131A26;border:1px solid #26354D;border-left:3px solid {accent};border-radius:10px;padding:7px 10px;">'
                f'<img src="{photo}" width="34" height="34" style="border-radius:50%;object-fit:cover;border:1.5px solid {accent};flex:none;">'
                f'<div style="min-width:0;flex:1;">'
                f'<div style="color:#F8FAFC;font-weight:800;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(disp_n)}</div>'
                f'<div style="color:#94A3B8;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(item.get("detail", ""))}</div>'
                f'</div>'
                f'{rating_badge_html}'
                f'<span style="white-space:nowrap;color:{item["status_color"]};font-size:10.5px;font-weight:800;">{html.escape(item["status_label"])}</span>'
                f'{minute_badge}'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:7px;margin-bottom:10px;">{"".join(cards)}</div>',
            unsafe_allow_html=True
        )

    home_bench, home_outgoing, home_injuries, home_has_roster, home_has_sub_data, home_has_injury_data = _team_substitution_status(home_events, match_squads.get(current_match['home_team']), ordered_players=home_players)
    away_bench, away_outgoing, away_injuries, away_has_roster, away_has_sub_data, away_has_injury_data = _team_substitution_status(away_events, match_squads.get(current_match['away_team']), ordered_players=away_players)
    col_home_status, col_away_status = st.columns(2, gap="medium")

    for status_col, team_name, is_home, substitutes, outgoing_players, injuries in [
        (col_home_status, current_match['home_team'], True, home_bench, home_outgoing, home_injuries),
        (col_away_status, current_match['away_team'], False, away_bench, away_outgoing, away_injuries),
    ]:
        team_color = '#38BDF8' if is_home else '#FF4D6D'
        with status_col:
            with st.container(border=True):
                st.markdown(f"<div style='color:{team_color};font-size:15px;font-weight:900;margin-bottom:8px;'>{'🔵' if is_home else '🔴'} {html.escape(team_name)}</div>", unsafe_allow_html=True)
                if substitutes:
                    st.markdown("<div style='color:#CBD5E1;font-size:12px;font-weight:800;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.04em;'>⬆ Dự bị & Vào sân</div>", unsafe_allow_html=True)
                    _render_status_grid(substitutes, is_home, team_name)
                if outgoing_players:
                    st.markdown("<div style='color:#CBD5E1;font-size:12px;font-weight:800;margin:6px 0;text-transform:uppercase;letter-spacing:0.04em;'>⬇ Cầu thủ rời sân</div>", unsafe_allow_html=True)
                    _render_status_grid(outgoing_players, is_home, team_name)
                if injuries:
                    st.markdown("<div style='color:#FBBF24;font-size:12px;font-weight:800;margin:6px 0;text-transform:uppercase;letter-spacing:0.04em;'>🩹 Chấn thương</div>", unsafe_allow_html=True)
                    _render_status_grid(injuries, is_home, team_name)
                if not substitutes and not outgoing_players and not injuries:
                    st.caption("Không ghi nhận dữ liệu thay người cho đội này.")

# ----------------------------------------------------------
# TAB 2: THỐNG KÊ TRẬN ĐẤU (FOTMOB / SOFASCORE BARS)
# ----------------------------------------------------------
with tab_stats:
    st.subheader("📊 Thống Kê So Sánh Chi Tiết Trận Đấu")

    home_passes = home_events[home_events['type_name'] == 'Pass'] if 'type_name' in home_events.columns else pd.DataFrame()
    away_passes = away_events[away_events['type_name'] == 'Pass'] if 'type_name' in away_events.columns else pd.DataFrame()
    home_summary = _pass_summary(home_passes)
    away_summary = _pass_summary(away_passes)

    h_col = "#38BDF8"
    a_col = "#FF4D6D"

    h_xg = _format_stat_val(home_source_stats, "xg", 2)
    a_xg = _format_stat_val(away_source_stats, "xg", 2)
    h_pos = _format_stat_val(home_source_stats, "possession_pct", 1, "%")
    a_pos = _format_stat_val(away_source_stats, "possession_pct", 1, "%")
    h_shots = _format_stat_val(home_source_stats, "shots")
    a_shots = _format_stat_val(away_source_stats, "shots")
    h_shots_on = _format_stat_val(home_source_stats, "shots_on_target")
    a_shots_on = _format_stat_val(away_source_stats, "shots_on_target")
    h_shots_off = _format_stat_val(home_source_stats, "shots_off_target")
    a_shots_off = _format_stat_val(away_source_stats, "shots_off_target")
    h_blocked = _format_stat_val(home_source_stats, "blocked_shots")
    a_blocked = _format_stat_val(away_source_stats, "blocked_shots")
    h_key_p = _format_stat_val(home_source_stats, "key_passes")
    a_key_p = _format_stat_val(away_source_stats, "key_passes")
    h_saves = _format_stat_val(home_source_stats, "goalkeeper_saves")
    a_saves = _format_stat_val(away_source_stats, "goalkeeper_saves")
    h_corners = _format_stat_val(home_source_stats, "corners")
    a_corners = _format_stat_val(away_source_stats, "corners")
    h_freekicks = _format_stat_val(home_source_stats, "free_kicks")
    a_freekicks = _format_stat_val(away_source_stats, "free_kicks")
    h_fouls = _format_stat_val(home_source_stats, "fouls")
    a_fouls = _format_stat_val(away_source_stats, "fouls")
    h_yellow = _format_stat_val(home_source_stats, "yellow_cards")
    a_yellow = _format_stat_val(away_source_stats, "yellow_cards")
    h_red = _format_stat_val(home_source_stats, "red_cards")
    a_red = _format_stat_val(away_source_stats, "red_cards")
    h_pass_acc = f"{home_summary['completed']:,} ({home_summary['accuracy']:.1f}%)" if home_summary['accuracy'] is not None else "Chưa có"
    a_pass_acc = f"{away_summary['completed']:,} ({away_summary['accuracy']:.1f}%)" if away_summary['accuracy'] is not None else "Chưa có"

    st.markdown("#### 📌 Tổng quan trận đấu")
    with st.container(horizontal=True, gap="small"):
        st.metric("Tỷ số · Nhà / khách", f"{current_match['home_score']} — {current_match['away_score']}", border=True)
        st.metric("xG · Nhà / khách", f"{h_xg} — {a_xg}", border=True)
        st.metric("Kiểm soát bóng · Nhà / khách", f"{h_pos} — {a_pos}", border=True)
        st.metric("Sút trúng đích · Nhà / khách", f"{h_shots_on} — {a_shots_on}", border=True)

    st.markdown("#### 📈 So sánh các chỉ số trọng tâm (Top Stats)")
    stat_bars_html = (
        f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 18px; margin-bottom: 14px; background: rgba(15, 23, 42, 0.85); border-radius: 10px; border: 1px solid #26354D;">'
        f'<span style="color: {h_col}; font-weight: 800; font-size: 15px;">🔵 {home_team_label}</span>'
        f'<span style="color: #94A3B8; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Đội dẫn trước được tô sáng</span>'
        f'<span style="color: {a_col}; font-weight: 800; font-size: 15px;">{away_team_label} 🔴</span>'
        f'</div>'
    )
    stat_bars_html += render_fotmob_stat_bar("Bàn thắng kỳ vọng (xG)", h_xg if h_xg != "Chưa có dữ liệu" else 1, a_xg if a_xg != "Chưa có dữ liệu" else 1, h_xg, a_xg, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Tỷ lệ kiểm soát bóng (%)", h_pos.replace("%", "") if h_pos != "Chưa có dữ liệu" else 50, a_pos.replace("%", "") if a_pos != "Chưa có dữ liệu" else 50, h_pos, a_pos, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Tổng số cú sút", h_shots if h_shots != "Chưa có dữ liệu" else 1, a_shots if a_shots != "Chưa có dữ liệu" else 1, h_shots, a_shots, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Sút trúng đích", h_shots_on if h_shots_on != "Chưa có dữ liệu" else 1, a_shots_on if a_shots_on != "Chưa có dữ liệu" else 1, h_shots_on, a_shots_on, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Chuyền chính xác", home_summary['completed'] or 1, away_summary['completed'] or 1, h_pass_acc, a_pass_acc, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Thủ môn cứu thua", h_saves if h_saves != "Chưa có dữ liệu" else 1, a_saves if a_saves != "Chưa có dữ liệu" else 1, h_saves, a_saves, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Phạt góc", h_corners if h_corners != "Chưa có dữ liệu" else 1, a_corners if a_corners != "Chưa có dữ liệu" else 1, h_corners, a_corners, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Phạm lỗi", h_fouls if h_fouls != "Chưa có dữ liệu" else 1, a_fouls if a_fouls != "Chưa có dữ liệu" else 1, h_fouls, a_fouls, h_col, a_col)
    stat_bars_html += render_fotmob_stat_bar("Thẻ vàng", h_yellow if h_yellow != "Chưa có dữ liệu" else 0, a_yellow if a_yellow != "Chưa có dữ liệu" else 0, h_yellow, a_yellow, h_col, a_col)

    with st.container(border=True):
        st.markdown(stat_bars_html, unsafe_allow_html=True)

    st.markdown("#### 🔍 Phân Tích Chuyên Sâu Theo Nhóm Chỉ Số (FotMob Pro Breakdown)")

    def _team_pair_count(summary, total_key, completed_key):
        total = summary[total_key]
        completed = summary[completed_key]
        if total is None:
            return "0/0"
        return f"{completed or 0:,}/{total:,}" if completed is not None else _display_count(total)

    col_stat_atk, col_stat_pass = st.columns(2, gap="medium")
    with col_stat_atk:
        with st.container(border=True):
            st.markdown("##### ⚡ Tấn Công, Dứt Điểm & Tình Huống Cố Định")
            atk_html = ""
            atk_html += render_fotmob_stat_bar("Cơ hội tạo ra (Key Passes)", h_key_p if h_key_p != "Chưa có dữ liệu" else (home_summary['key_passes'] or 0), a_key_p if a_key_p != "Chưa có dữ liệu" else (away_summary['key_passes'] or 0), h_key_p if h_key_p != "Chưa có dữ liệu" else str(home_summary['key_passes'] or 0), a_key_p if a_key_p != "Chưa có dữ liệu" else str(away_summary['key_passes'] or 0), h_col, a_col)
            atk_html += render_fotmob_stat_bar("Kiến tạo thành bàn (Assists)", home_summary['assists'] or 0, away_summary['assists'] or 0, _display_count(home_summary['assists'] or 0), _display_count(away_summary['assists'] or 0), h_col, a_col)
            atk_html += render_fotmob_stat_bar("Sút ra ngoài (Off Target)", h_shots_off if h_shots_off != "Chưa có dữ liệu" else 0, a_shots_off if a_shots_off != "Chưa có dữ liệu" else 0, h_shots_off, a_shots_off, h_col, a_col)
            atk_html += render_fotmob_stat_bar("Sút bị chặn (Blocked Shots)", h_blocked if h_blocked != "Chưa có dữ liệu" else 0, a_blocked if a_blocked != "Chưa có dữ liệu" else 0, h_blocked, a_blocked, h_col, a_col)
            atk_html += render_fotmob_stat_bar("Đá phạt trực tiếp / gián tiếp", h_freekicks if h_freekicks != "Chưa có dữ liệu" else 0, a_freekicks if a_freekicks != "Chưa có dữ liệu" else 0, h_freekicks, a_freekicks, h_col, a_col)
            atk_html += render_fotmob_stat_bar("Thẻ đỏ (Red Cards)", h_red if h_red != "Chưa có dữ liệu" else 0, a_red if a_red != "Chưa có dữ liệu" else 0, h_red, a_red, h_col, a_col)
            st.markdown(atk_html, unsafe_allow_html=True)

    with col_stat_pass:
        with st.container(border=True):
            st.markdown("##### 🎯 Tổ Chức Chuyền Bóng & Độ Đe Dọa (Expected Threat - xT)")
            pass_html = ""
            pass_html += render_fotmob_stat_bar("Tổng số đường chuyền", home_summary['attempted'] or 0, away_summary['attempted'] or 0, _display_count(home_summary['attempted'] or 0), _display_count(away_summary['attempted'] or 0), h_col, a_col)
            pass_html += render_fotmob_stat_bar("Chuyền tịnh tiến (Thành công/Tổng)", home_summary['progressive_completed'] or 0, away_summary['progressive_completed'] or 0, _team_pair_count(home_summary, 'progressive', 'progressive_completed'), _team_pair_count(away_summary, 'progressive', 'progressive_completed'), h_col, a_col)
            pass_html += render_fotmob_stat_bar("Tạt bóng (Thành công/Tổng)", home_summary['crosses_completed'] or 0, away_summary['crosses_completed'] or 0, _team_pair_count(home_summary, 'crosses', 'crosses_completed'), _team_pair_count(away_summary, 'crosses', 'crosses_completed'), h_col, a_col)
            pass_html += render_fotmob_stat_bar("Chọc khe (Thành công/Tổng)", home_summary['throughballs_completed'] or 0, away_summary['throughballs_completed'] or 0, _team_pair_count(home_summary, 'throughballs', 'throughballs_completed'), _team_pair_count(away_summary, 'throughballs', 'throughballs_completed'), h_col, a_col)
            pass_html += render_fotmob_stat_bar("Chuyển cánh (Switches of Play)", home_summary['switches'] or 0, away_summary['switches'] or 0, _display_count(home_summary['switches'] or 0), _display_count(away_summary['switches'] or 0), h_col, a_col)
            pass_html += render_fotmob_stat_bar("Chỉ số đe dọa từ chuyền (xT)", max(0.01, home_summary['xt'] or 0.0), max(0.01, away_summary['xt'] or 0.0), _display_decimal(home_summary['xt'], 3, signed=True), _display_decimal(away_summary['xt'], 3, signed=True), h_col, a_col)
            st.markdown(pass_html, unsafe_allow_html=True)

    render_attack_momentum_and_defense(match_events, current_match, home_source_stats, away_source_stats)

# ----------------------------------------------------------
# TAB 3: BẢN ĐỒ SÚT BÓNG (SHOT MAP)
# ----------------------------------------------------------
with tab_shotmap:
    st.subheader("🎯 Bản Đồ Dứt Điểm Toàn Trận & Đường Đua xG (Interactive Shot Map)")
    with st.container(border=True):
        shot_filter = st.selectbox(
            "🎛️ Lọc cú sút trên bản đồ:",
            ["Tất cả cú sút (Cả 2 đội)", f"🔵 {current_match['home_team']} · tấn công sang phải →", f"🔴 {current_match['away_team']} · tấn công sang trái ←", "⚽ Chỉ bàn thắng (Goals)", "🎯 Trúng đích (On Target)"],
            key=f"shot_map_filter_{current_match['id']}"
        )
    render_shot_map(match_events, current_match['home_team'], current_match['away_team'], shot_filter, current_match=current_match)

# ----------------------------------------------------------
# TAB 4: PHÂN TÍCH CẦU THỦ (HEATMAP & PASS MAP)
# ----------------------------------------------------------
with tab_player:
    st.subheader("🔥 Phân Tích Chuyên Sâu Cầu Thủ: Heatmap Hoạt Động & Bản Đồ Chuyền")

    all_match_players = [p['name'] for p in all_selectable_players] if all_selectable_players else (home_players + away_players)
    if not all_match_players:
        all_match_players = ["Cầu thủ"]
    player_meta_lookup = {p['name']: p for p in all_selectable_players}

    with st.container(border=True):
        selected_player = st.selectbox(
            "🔍 Chọn cầu thủ cần phân tích:",
            all_match_players,
            index=0,
            format_func=lambda n: (
                f"{'🔄 ' if player_meta_lookup[n].get('is_sub') else ''}{player_meta_lookup[n]['display_name']} "
                f"({player_meta_lookup[n].get('team_name', '')} • {'Dự bị' if player_meta_lookup[n].get('is_sub') else player_meta_lookup[n]['pos']} • {player_meta_lookup[n]['rating']}★)"
            ) if n in player_meta_lookup else _player_display_name(n),
            key=f"player_analytics_select_{current_match['id']}"
        )

    p_info = player_meta_lookup.get(selected_player)
    p_events = match_events[match_events['player_name'] == selected_player]
    p_passes = p_events[p_events['type_name'] == 'Pass'] if 'type_name' in p_events.columns else pd.DataFrame()
    player_team_values = p_events['team_name'].dropna() if 'team_name' in p_events.columns else pd.Series(dtype="object")
    selected_team = str(player_team_values.iloc[0]) if not player_team_values.empty else (p_info.get('team_name', '') if p_info else "")
    is_sel_home = (selected_team == current_match['home_team'])
    player_position_values = p_events['position_name'].dropna() if 'position_name' in p_events.columns else pd.Series(dtype="object")
    selected_position = p_info['pos'] if p_info else (str(player_position_values.iloc[0]) if not player_position_values.empty else "Cầu thủ")
    selected_avatar = p_info['photo'] if p_info else get_player_avatar(selected_player, is_sel_home, selected_team)
    selected_rating = p_info['rating'] if p_info else calculate_player_rating(p_events)[0]
    pass_stats = _pass_summary(p_passes)

    sel_r_col = "#00D66C" if selected_rating >= 7.5 else ("#FF9800" if selected_rating >= 6.5 else "#EF4444")
    sel_t_col = "#38BDF8" if is_sel_home else "#FF4D6D"
    sel_goals = p_info.get('goals_count', 0) if p_info else 0
    sel_assists = p_info.get('assists_count', 0) if p_info else (pass_stats['assists'] or 0)
    sel_touches = len(p_events)

    st.markdown(
        f'<div style="background: linear-gradient(135deg, #151E2E 0%, #101724 100%); border: 1.5px solid #26354D; border-left: 4px solid {sel_t_col}; border-radius: 14px; padding: 16px 22px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">'
        f'<div style="display: flex; align-items: center; gap: 16px;">'
        f'<img src="{selected_avatar}" width="72" height="72" style="border-radius: 50%; object-fit: cover; border: 3px solid {sel_r_col};">'
        f'<div>'
        f'<div style="font-size: 22px; font-weight: 900; color: #FFFFFF;">{html.escape(_player_display_name(selected_player))}</div>'
        f'<div style="color: #94A3B8; font-size: 13.5px; font-weight: 700; margin-top: 4px;">'
        f'<span style="color: {sel_t_col};">{"🔵" if is_sel_home else "🔴"} {html.escape(selected_team)}</span> &nbsp;•&nbsp; Vị trí: <b style="color: #E2E8F0;">{html.escape(selected_position)}</b> &nbsp;•&nbsp; Hoạt động: <b style="color: #E2E8F0;">{sel_touches} sự kiện</b>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; align-items: center; gap: 14px;">'
        f'<div style="text-align: right;">'
        f'<div style="color: #94A3B8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;">Sofascore Rating</div>'
        f'<div style="color: #CBD5E1; font-size: 12px; margin-top: 2px;">⚽ {sel_goals} Bàn &nbsp;|&nbsp; 👟 {sel_assists} Kiến tạo</div>'
        f'</div>'
        f'<div style="background: {sel_r_col}; color: #000000; font-size: 24px; font-weight: 900; padding: 8px 16px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);">{selected_rating:.1f}★</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    activity_events = p_events[p_events['location_x'].notna() & p_events['location_y'].notna()]
    pass_accuracy_text = "Chưa có" if pass_stats['accuracy'] is None else f"{pass_stats['completed']:,}/{pass_stats['attempted']:,} ({pass_stats['accuracy']:.1f}%)"
    progressive_text = "Chưa có" if pass_stats['progressive'] is None else f"{pass_stats['progressive_completed'] or 0:,}/{pass_stats['progressive']:,}"

    with st.container(horizontal=True, gap="small"):
        st.metric("Điểm SofaScore", f"{selected_rating:.1f} ★", border=True)
        st.metric("Chuyền chính xác", pass_accuracy_text, border=True)
        st.metric("xT từ đường chuyền", _display_decimal(pass_stats['xt'], 3, signed=True), border=True)
        st.metric("Key Passes (Tạo cơ hội)", _display_count(pass_stats['key_passes'] or 0), border=True)
    with st.container(horizontal=True, gap="small"):
        st.metric("Chuyền tịnh tiến", progressive_text, border=True)
        st.metric("Kiến tạo (Assists)", _display_count(pass_stats['assists'] or 0), border=True)
        st.metric("Tạt bóng (Crosses)", _display_count(pass_stats['crosses'] or 0), border=True)
        st.metric("Chọc khe (Throughballs)", _display_count(pass_stats['throughballs'] or 0), border=True)

    st.markdown("#### 📍 Bản đồ không gian hoạt động & đường chuyền (Hướng tấn công: Trái ➔ Phải)")
    col_heat, col_pass_map = st.columns([1, 1], gap="medium")

    with col_heat:
        with st.container(border=True):
            st.markdown(f"##### 🔥 Heatmap Hoạt Động — {_player_display_name(selected_player)}")
            if {"location_x", "location_y"}.issubset(activity_events.columns):
                coords_df = activity_events[["location_x", "location_y"]].copy()
                coords_df["location_x"] = pd.to_numeric(coords_df["location_x"], errors="coerce")
                coords_df["location_y"] = pd.to_numeric(coords_df["location_y"], errors="coerce")
                coords_df = coords_df.dropna(subset=["location_x", "location_y"])
                coords_df["location_x"] = coords_df["location_x"].clip(0, 120)
                coords_df["location_y"] = coords_df["location_y"].clip(0, 80)
            else:
                coords_df = pd.DataFrame(columns=["location_x", "location_y"])

            if not coords_df.empty:
                fig_heat = go.Figure()
                for i in range(12):
                    sc = "#121B14" if i % 2 == 0 else "#152217"
                    fig_heat.add_shape(type="rect", x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80, fillcolor=sc, line=dict(width=0), layer="below")

                sofascore_scale = [
                    [0.0, "rgba(18, 27, 20, 0.0)"],
                    [0.12, "rgba(16, 185, 129, 0.26)"],
                    [0.35, "rgba(132, 204, 22, 0.54)"],
                    [0.60, "rgba(250, 204, 21, 0.78)"],
                    [0.82, "rgba(249, 115, 22, 0.90)"],
                    [1.0, "rgba(239, 68, 68, 0.96)"],
                ]
                if len(coords_df) >= 3:
                    fig_heat.add_trace(go.Histogram2dContour(
                        x=coords_df["location_x"], y=coords_df["location_y"],
                        colorscale=sofascore_scale, reversescale=False,
                        ncontours=20, showscale=False,
                        contours=dict(coloring="fill", showlines=False),
                        hoverinfo="skip"
                    ))
                fig_heat.add_trace(go.Scatter(
                    x=coords_df["location_x"], y=coords_df["location_y"],
                    mode="markers",
                    marker=dict(size=5.5, color="rgba(255, 255, 255, 0.45)", line=dict(color="#0B0F19", width=0.5)),
                    name="Điểm chạm bóng",
                    hovertemplate="Tọa độ: (%{x:.1f}, %{y:.1f})<extra></extra>"
                ))
                pitch_line = dict(color="rgba(255, 255, 255, 0.48)", width=1.6)
                fig_heat.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=pitch_line, layer="above")
                fig_heat.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=pitch_line, layer="above")
                fig_heat.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=pitch_line, layer="above")
                fig_heat.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=pitch_line, layer="above")
                fig_heat.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=pitch_line, layer="above")
                fig_heat.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=pitch_line, layer="above")
                fig_heat.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=pitch_line, layer="above")
                fig_heat.add_annotation(
                    x=60, y=76.5, text=f"HƯỚNG TẤN CÔNG CỦA {selected_team.upper()} ➔",
                    showarrow=False, font=dict(size=11, color="#A7F3D0", family="Arial Black, sans-serif"),
                    bgcolor="rgba(15, 23, 42, 0.78)", borderpad=3
                )
                fig_heat.update_layout(
                    xaxis=dict(range=[-3, 123], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                    yaxis=dict(range=[83, -3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True),
                    height=450, margin=dict(l=5, r=5, t=10, b=10),
                    plot_bgcolor="#121B14", paper_bgcolor="#131A26",
                    font=dict(color="#E2E8F0"), showlegend=False
                )
                st.plotly_chart(fig_heat, width="stretch")
            else:
                st.info("Không có tọa độ hợp lệ cho cầu thủ này.")

    with col_pass_map:
        with st.container(border=True):
            st.markdown(f"##### 🎯 Bản Đồ Đường Chuyền — {_player_display_name(selected_player)}")
            success_mask = _pass_success_mask(p_passes)
            map_passes = p_passes.copy()
            if success_mask is not None:
                map_passes["_pass_complete"] = success_mask.reindex(map_passes.index).fillna(False).astype(bool)
            else:
                map_passes["_pass_complete"] = True

            required_coordinates = {"location_x", "location_y", "pass_end_location_x", "pass_end_location_y"}
            if not required_coordinates.issubset(map_passes.columns):
                st.info("Nguồn chưa có đủ tọa độ đầu và cuối để vẽ đường chuyền.")
            else:
                for column in required_coordinates:
                    map_passes[column] = pd.to_numeric(map_passes[column], errors="coerce")
                map_passes = map_passes.dropna(subset=list(required_coordinates))
                map_passes["location_x"] = map_passes["location_x"].clip(0, 120)
                map_passes["pass_end_location_x"] = map_passes["pass_end_location_x"].clip(0, 120)
                map_passes["location_y"] = map_passes["location_y"].clip(0, 80)
                map_passes["pass_end_location_y"] = map_passes["pass_end_location_y"].clip(0, 80)

                if map_passes.empty:
                    st.info("Không có dữ liệu đường chuyền.")
                else:
                    fig_pass = go.Figure()
                    for i in range(12):
                        sc = "#121B14" if i % 2 == 0 else "#152217"
                        fig_pass.add_shape(type="rect", x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80, fillcolor=sc, line=dict(width=0), layer="below")

                    key_mask = (
                        pd.to_numeric(map_passes.get("pass_goal_assist", 0), errors="coerce").fillna(0).eq(1) |
                        pd.to_numeric(map_passes.get("pass_shot_assist", 0), errors="coerce").fillna(0).eq(1)
                    ) if ("pass_goal_assist" in map_passes.columns or "pass_shot_assist" in map_passes.columns) else pd.Series(False, index=map_passes.index)

                    line_groups = [
                        ("Chuyền chính xác", map_passes[map_passes["_pass_complete"] & ~key_mask], "#00E676", 1.7),
                        ("Key Pass / Kiến tạo", map_passes[key_mask], "#FACC15", 2.8),
                        ("Chuyền hỏng", map_passes[~map_passes["_pass_complete"] & ~key_mask], "#FF4D6D", 1.4),
                    ]
                    for trace_name, rows, color, lw in line_groups:
                        if rows.empty:
                            continue
                        line_x, line_y = [], []
                        for _, row in rows.iterrows():
                            line_x.extend([row["location_x"], row["pass_end_location_x"], None])
                            line_y.extend([row["location_y"], row["pass_end_location_y"], None])
                        fig_pass.add_trace(go.Scatter(
                            x=line_x, y=line_y, mode="lines", name=f"{trace_name} ({len(rows)})",
                            line=dict(color=color, width=lw), hoverinfo="skip"
                        ))

                    marker_colors = [
                        "#FACC15" if is_k else ("#00E676" if is_c else "#FF4D6D")
                        for is_c, is_k in zip(map_passes["_pass_complete"].tolist(), key_mask.tolist())
                    ]
                    hover_text = []
                    for _, row in map_passes.iterrows():
                        minute_text = f"Phút {int(row['minute'])}′" if 'minute' in row and pd.notna(row['minute']) else ""
                        recipient = _player_display_name(str(row.get("pass_recipient_name", ""))) if pd.notna(row.get("pass_recipient_name")) and str(row.get("pass_recipient_name")).strip() else "Đồng đội"
                        result_text = "Chính xác" if pd.notna(row["_pass_complete"]) and bool(row["_pass_complete"]) else "Chuyền hỏng"
                        hover_text.append(f"<b>{minute_text}</b><br>Đến: <b>{recipient}</b><br>Kết quả: <b>{result_text}</b>")
                    fig_pass.add_trace(go.Scatter(
                        x=map_passes["pass_end_location_x"], y=map_passes["pass_end_location_y"], mode="markers",
                        marker=dict(size=7.5, color=marker_colors, symbol="circle", line=dict(color="#0B0D10", width=1)),
                        text=hover_text, hovertemplate="%{text}<extra></extra>", name="Điểm đến"
                    ))
                    pitch_line = dict(color="rgba(255, 255, 255, 0.48)", width=1.6)
                    fig_pass.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=pitch_line, layer="below")
                    fig_pass.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=pitch_line)
                    fig_pass.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=pitch_line)
                    fig_pass.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=pitch_line)
                    fig_pass.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=pitch_line)
                    fig_pass.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=pitch_line)
                    fig_pass.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=pitch_line)
                    fig_pass.add_annotation(
                        x=60, y=76.5, text=f"HƯỚNG TẤN CÔNG CỦA {selected_team.upper()} ➔",
                        showarrow=False, font=dict(size=11, color="#A7F3D0", family="Arial Black, sans-serif"),
                        bgcolor="rgba(15, 23, 42, 0.78)", borderpad=3
                    )
                    fig_pass.update_layout(
                        xaxis=dict(range=[-3, 123], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                        yaxis=dict(range=[83, -3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True),
                        height=450, margin=dict(l=5, r=5, t=10, b=10),
                        plot_bgcolor="#121B14", paper_bgcolor="#131A26",
                        font=dict(color="#E2E8F0"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, bgcolor="rgba(19, 26, 38, 0.9)")
                    )
                    st.plotly_chart(fig_pass, width="stretch")

# ----------------------------------------------------------
# TAB 5: MẠNG LƯỚI PHỐI HỢP CHUYỀN BÓNG (PASS NETWORK)
# ----------------------------------------------------------
with tab_network:
    st.subheader("🕸️ Mạng Lưới Phối Hợp Chuyền Bóng Chiến Thuật (Passing Network)")
    match_passes = match_events[match_events['type_name'] == 'Pass'] if 'type_name' in match_events.columns else pd.DataFrame()

    with st.container(border=True):
        network_col, threshold_col = st.columns([2, 1])
        with network_col:
            network_team = st.selectbox(
                "🎛️ Chọn đội phân tích mạng lưới:",
                [current_match['home_team'], current_match['away_team']],
                format_func=lambda t: f"{'🔵' if t == current_match['home_team'] else '🔴'} {t}",
                key=f"network_team_{current_match['id']}"
            )
        with threshold_col:
            min_link_count = st.slider("Ngưỡng số đường chuyền tối thiểu:", min_value=1, max_value=8, value=2, key=f"network_min_{current_match['id']}")

    is_net_home = (network_team == current_match['home_team'])
    net_node_color = "#38BDF8" if is_net_home else "#FF4D6D"
    net_arrow_rgba = "rgba(56, 189, 248, 0.55)" if is_net_home else "rgba(255, 77, 109, 0.55)"
    net_line_rgba = "rgba(56, 189, 248, 0.28)" if is_net_home else "rgba(255, 77, 109, 0.28)"

    network_team_passes = match_passes[match_passes['team_name'] == network_team].copy() if 'team_name' in match_passes.columns else pd.DataFrame()
    recipient_available = 'pass_recipient_name' in network_team_passes.columns
    position_columns_available = {"player_name", "location_x", "location_y", "pass_end_location_x", "pass_end_location_y", "pass_recipient_name"}.issubset(network_team_passes.columns)

    if not recipient_available or not position_columns_available:
        st.info("Nguồn chưa có đủ người nhận chuyền và tọa độ để tạo mạng lưới phối hợp.")
    else:
        network_success = _pass_success_mask(network_team_passes)
        network_passes = network_team_passes.loc[network_success].copy() if network_success is not None else network_team_passes.copy()
        network_passes = network_passes.dropna(subset=["player_name", "pass_recipient_name"])
        network_passes = network_passes[network_passes['player_name'] != network_passes['pass_recipient_name']]
        edge_counts = network_passes.groupby(["player_name", "pass_recipient_name"]).size().reset_index(name="Số chuyền")

        starts = network_team_passes[["player_name", "location_x", "location_y"]].rename(columns={"player_name": "Cầu thủ", "location_x": "x", "location_y": "y"})
        ends = network_team_passes[["pass_recipient_name", "pass_end_location_x", "pass_end_location_y"]].rename(columns={"pass_recipient_name": "Cầu thủ", "pass_end_location_x": "x", "pass_end_location_y": "y"})
        player_positions = pd.concat([starts, ends], ignore_index=True)
        player_positions["x"] = pd.to_numeric(player_positions["x"], errors="coerce")
        player_positions["y"] = pd.to_numeric(player_positions["y"], errors="coerce")
        player_positions = player_positions.dropna(subset=["Cầu thủ", "x", "y"])
        player_positions["x"] = player_positions["x"].clip(0, 120)
        player_positions["y"] = player_positions["y"].clip(0, 80)
        player_positions = player_positions.groupby("Cầu thủ")[["x", "y"]].mean()

        # Tách nhẹ các nút vị trí trung bình quá sát nhau để không đè chữ trên sơ đồ mạng lưới
        pos_names_list = player_positions.index.tolist()
        for _pass_iter in range(5):
            moved_node = False
            for i_n in range(len(pos_names_list)):
                for j_n in range(i_n + 1, len(pos_names_list)):
                    n1, n2 = pos_names_list[i_n], pos_names_list[j_n]
                    dx = float(player_positions.loc[n2, "x"] - player_positions.loc[n1, "x"])
                    dy = float(player_positions.loc[n2, "y"] - player_positions.loc[n1, "y"])
                    dist = np.hypot(dx, dy)
                    if dist < 5.5:
                        ux, uy = (0.0, 1.0) if dist < 1e-3 else (dx / dist, dy / dist)
                        shift = 0.5 * (5.5 - dist)
                        player_positions.loc[n1, "x"] = np.clip(player_positions.loc[n1, "x"] - ux * shift, 4, 116)
                        player_positions.loc[n1, "y"] = np.clip(player_positions.loc[n1, "y"] - uy * shift, 5, 75)
                        player_positions.loc[n2, "x"] = np.clip(player_positions.loc[n2, "x"] + ux * shift, 4, 116)
                        player_positions.loc[n2, "y"] = np.clip(player_positions.loc[n2, "y"] + uy * shift, 5, 75)
                        moved_node = True
            if not moved_node:
                break

        visible_edges = edge_counts[edge_counts["Số chuyền"] >= min_link_count].sort_values("Số chuyền", ascending=False).head(60)
        visible_edges = visible_edges[
            visible_edges["player_name"].isin(player_positions.index) &
            visible_edges["pass_recipient_name"].isin(player_positions.index)
        ]

        if visible_edges.empty:
            st.info("Không có liên kết nào đạt ngưỡng hiện tại trong trận này.")
        else:
            outgoing_counts = edge_counts.groupby("player_name")["Số chuyền"].sum().to_dict()
            incoming_counts = edge_counts.groupby("pass_recipient_name")["Số chuyền"].sum().to_dict()
            hub_player = max(player_positions.index.tolist(), key=lambda n: outgoing_counts.get(n, 0) + incoming_counts.get(n, 0)) if not player_positions.empty else ""
            hub_total = outgoing_counts.get(hub_player, 0) + incoming_counts.get(hub_player, 0) if hub_player else 0
            top_edge_row = visible_edges.iloc[0]
            top_pair_str = f"{_player_display_name(str(top_edge_row['player_name']), pitch=True)} ➔ {_player_display_name(str(top_edge_row['pass_recipient_name']), pitch=True)}"

            k_net1, k_net2, k_net3, k_net4 = st.columns(4, gap="small")
            k_net1.metric("Chuyền phối hợp thành công", f"{len(network_passes):,}", border=True)
            k_net2.metric(f"Liên kết (≥ {min_link_count} chuyền)", f"{len(visible_edges):,} cặp", border=True)
            k_net3.metric("Trạm luân chuyển chính (Hub)", f"{_player_display_name(hub_player)} ({hub_total})" if hub_player else "—", border=True)
            k_net4.metric("Cặp phối hợp ăn ý nhất", f"{top_pair_str} ({int(top_edge_row['Số chuyền'])})", border=True)

            col_net_pitch, col_net_pairs = st.columns([1.45, 1.0], gap="medium")
            with col_net_pitch:
                with st.container(border=True):
                    st.markdown(f"##### 🕸️ Sơ Đồ Vị Trí Trung Bình & Mạng Lưới Chuyền — {network_team}")
                    fig_network = go.Figure()
                    for i in range(12):
                        sc = "#121B14" if i % 2 == 0 else "#152217"
                        fig_network.add_shape(type="rect", x0=i * 10, y0=0, x1=(i + 1) * 10, y1=80, fillcolor=sc, line=dict(width=0), layer="below")

                    line_x, line_y = [], []
                    for _, edge in visible_edges.iterrows():
                        source_position = player_positions.loc[edge["player_name"]]
                        target_position = player_positions.loc[edge["pass_recipient_name"]]
                        line_x.extend([source_position["x"], target_position["x"], None])
                        line_y.extend([source_position["y"], target_position["y"], None])
                        fig_network.add_annotation(
                            x=target_position["x"], y=target_position["y"],
                            ax=source_position["x"], ay=source_position["y"],
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=0.85,
                            arrowwidth=min(4.5, 1.1 + float(edge["Số chuyền"]) / 3.5),
                            arrowcolor=net_arrow_rgba, text=""
                        )
                    fig_network.add_trace(go.Scatter(
                        x=line_x, y=line_y, mode="lines",
                        line=dict(color=net_line_rgba, width=1.2),
                        hoverinfo="skip", showlegend=False
                    ))

                    player_names = [p for p in player_positions.index.tolist() if str(p).strip() and str(p).strip().lower() != 'nan']
                    hover_text = [
                        f"<b>{_player_display_name(name)}</b><br>Chuyền đi: <b>{outgoing_counts.get(name, 0):,}</b><br>Chuyền nhận: <b>{incoming_counts.get(name, 0):,}</b>"
                        for name in player_names
                    ]
                    node_sizes = [16 + min(20, 3.4 * np.log1p(outgoing_counts.get(name, 0))) for name in player_names]
                    fig_network.add_trace(go.Scatter(
                        x=[player_positions.loc[name, "x"] for name in player_names],
                        y=[player_positions.loc[name, "y"] for name in player_names],
                        mode="markers+text", text=[f"<b>{_player_display_name(name, pitch=True)}</b>" for name in player_names],
                        textposition="top center", textfont=dict(color="#FFFFFF", size=11),
                        marker=dict(size=node_sizes, color=net_node_color, line=dict(color="#FFFFFF", width=2), opacity=0.95),
                        hovertext=hover_text, hovertemplate="%{hovertext}<extra></extra>", name="Cầu thủ"
                    ))
                    pitch_line = dict(color="rgba(255, 255, 255, 0.45)", width=1.6)
                    fig_network.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=pitch_line, layer="below")
                    fig_network.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=pitch_line)
                    fig_network.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=pitch_line)
                    fig_network.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=pitch_line)
                    fig_network.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=pitch_line)
                    fig_network.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=pitch_line)
                    fig_network.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=pitch_line)
                    fig_network.add_annotation(
                        x=60, y=76.5, text=f"HƯỚNG TẤN CÔNG CỦA {network_team.upper()} ➔",
                        showarrow=False, font=dict(size=11, color="#F8FAFC", family="Arial Black, sans-serif"),
                        bgcolor="rgba(15, 23, 42, 0.8)", borderpad=4
                    )
                    fig_network.update_layout(
                        xaxis=dict(range=[-3, 123], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                        yaxis=dict(range=[83, -3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True),
                        height=520, margin=dict(l=5, r=5, t=10, b=10),
                        plot_bgcolor="#121B14", paper_bgcolor="#131A26",
                        font=dict(color="#E2E8F0"), showlegend=False
                    )
                    st.plotly_chart(fig_network, width="stretch")
                    st.caption(f"Đang hiển thị {len(visible_edges)} liên kết có từ {min_link_count} đường chuyền chính xác trở lên. Kích thước nút tỷ lệ với khối lượng luân chuyển bóng.")

            with col_net_pairs:
                with st.container(border=True):
                    st.markdown("##### 🔗 Top 10 Cặp Phối Hợp Ăn Ý Nhất")
                    top_pairs = visible_edges.head(10)
                    max_pair_cnt = int(top_pairs["Số chuyền"].max()) if not top_pairs.empty else 1
                    pairs_html = '<div style="display: flex; flex-direction: column; gap: 8px; max-height: 520px; overflow-y: auto; padding-right: 4px;">'
                    for rank_idx, (_, pr) in enumerate(top_pairs.iterrows(), start=1):
                        p1_raw = str(pr["player_name"])
                        p2_raw = str(pr["pass_recipient_name"])
                        p1_disp = html.escape(_player_display_name(p1_raw))
                        p2_disp = html.escape(_player_display_name(p2_raw))
                        p1_av = get_player_avatar(p1_raw, is_net_home, network_team)
                        p2_av = get_player_avatar(p2_raw, is_net_home, network_team)
                        cnt_val = int(pr["Số chuyền"])
                        bar_w = max(12, round(cnt_val / max_pair_cnt * 100))
                        pairs_html += (
                            f'<div style="background: #131A26; border: 1px solid #26354D; border-radius: 10px; padding: 8px 12px;">'
                            f'<div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px;">'
                            f'<div style="display: flex; align-items: center; gap: 7px; min-width: 0; flex: 1;">'
                            f'<span style="color: #64748B; font-weight: 900; font-size: 12px; width: 18px;">#{rank_idx}</span>'
                            f'<img src="{p1_av}" width="30" height="30" style="border-radius: 50%; object-fit: cover; border: 1.5px solid {net_node_color};">'
                            f'<span style="color: #F8FAFC; font-weight: 800; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p1_disp}</span>'
                            f'<span style="color: {net_node_color}; font-weight: 900; font-size: 13px;">➔</span>'
                            f'<img src="{p2_av}" width="30" height="30" style="border-radius: 50%; object-fit: cover; border: 1.5px solid {net_node_color};">'
                            f'<span style="color: #F8FAFC; font-weight: 800; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p2_disp}</span>'
                            f'</div>'
                            f'<span style="background: rgba(56, 189, 248, 0.16); border: 1px solid {net_node_color}; color: #FFFFFF; font-weight: 900; font-size: 12.5px; padding: 2px 9px; border-radius: 999px; white-space: nowrap;">{cnt_val} chuyền</span>'
                            f'</div>'
                            f'<div style="height: 5px; background: #1E293B; border-radius: 999px; overflow: hidden;">'
                            f'<div style="width: {bar_w}%; height: 100%; background: {net_node_color}; border-radius: 999px;"></div>'
                            f'</div>'
                            f'</div>'
                        )
                    pairs_html += '</div>'
                    st.markdown(pairs_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 6: AI MONTE CARLO DỰ ĐOÁN, BÌNH LUẬN TRỰC TIẾP & BẢN TIN
# ----------------------------------------------------------
with tab_ai_news:
    render_match_ai_timeline_news_tab(match_events, current_match, all_selectable_players)

