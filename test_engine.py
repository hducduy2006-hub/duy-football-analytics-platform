import json
import os
import pandas as pd
import numpy as np

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
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0

def parse_statsbomb_events_json(raw_events, match_id=None):
    valid_periods = {1, 2, 3, 4}
    
    # 1. Detect teams & lineups from Starting XI
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
    
    # 2. Extract match events & goals
    rows = []
    goals_info = []
    home_score = 0
    away_score = 0
    
    for e in raw_events:
        t_name = (e.get('type') or {}).get('name', '')
        team = (e.get('team') or {}).get('name', '')
        player = (e.get('player') or {}).get('name', '')
        pos = (e.get('position') or {}).get('name', '')
        period = e.get('period', 1)
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
        p_type = (pass_data.get('type') or {}).get('name', '')
        
        shot_data = e.get('shot') or {}
        s_outcome = (shot_data.get('outcome') or {}).get('name', '')
        s_xg = shot_data.get('statsbomb_xg')
        if t_name == 'Shot' and s_xg is None and x is not None and y is not None:
            s_xg = estimate_xg(x, y)
            
        if t_name == 'Shot' and s_outcome == 'Goal' and period in valid_periods:
            goals_info.append({"player": player, "minute": minute, "team": team})
            if team == home_team:
                home_score += 1
            elif team == away_team:
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
            'under_pressure': 1 if e.get('under_pressure') else 0,
            'possession': e.get('possession'),
            'possession_team': (e.get('possession_team') or {}).get('name', ''),
        }
        rows.append(row)
        
    events_df = pd.DataFrame(rows)
    
    # 3. Compute match stats
    stats_dict = {t: {
        'possession_seconds': 0.0, 'shots': 0, 'shots_on_target': 0, 'shots_off_target': 0,
        'blocked_shots': 0, 'xg': 0.0, 'key_passes': 0, 'goalkeeper_saves': 0, 'corners': 0,
        'fouls': 0, 'free_kicks': 0, 'yellow_cards': 0, 'red_cards': 0, 'goals': 0,
        'passes': 0, 'passes_completed': 0
    } for t in [home_team, away_team]}
    
    # Possession
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
        'goals': goals_info,
        'events_df': events_df,
        'stats_dict': stats_dict,
        'starting_lineups': starting_lineups
    }

if __name__ == '__main__':
    with open('sample_data/wc2022_final_arg_fra.json', 'r', encoding='utf-8') as f:
        evs = json.load(f)
    res = parse_statsbomb_events_json(evs, 3869685)
    print("Match:", res['home_team'], res['home_score'], '-', res['away_score'], res['away_team'])
    print("Stats Home:", res['stats_dict'][res['home_team']])
    print("Stats Away:", res['stats_dict'][res['away_team']])
    print("Starting XI count:", len(res['starting_lineups'][res['home_team']]['players']))
