"""
Example: Using Google Custom Search API for Simple Image Search
---------------------------------------------------------------
This example integrates a Google Custom Search (CSE) request when the user
submits a game without providing an image URL. If the search succeeds, we
assign the top-result image to the new game. Otherwise, we fall back to
a placeholder image URL.

Prerequisites:
--------------
1) You must have a Google Cloud project with the Custom Search JSON API enabled:
   https://developers.google.com/custom-search/v1/introduction

2) Set up a custom search engine (CSE) in your Google account and obtain:
   - A "cx" (custom search engine ID).
   - An API key.

3) Install the 'requests' package if not already present:
   pip install requests

4) Replace 'YOUR_GOOGLE_API_KEY' and 'YOUR_CUSTOM_SEARCH_CX' in the code with
   your real credentials.

Usage:
------
When a user submits a game with no image_url, we'll call `fetch_google_image_url(title)`
to query the Google Custom Search API. If a relevant image is found, it will be used.
Otherwise, we use the placeholder approach.

All other logic remains as in the existing app structure.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from collections import Counter
import random

import requests  # <-- Needed for the Google CSE HTTP request
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# 1) Configuration for Google Custom Search
# ------------------------------------------------------------------------------
GOOGLE_API_KEY = "AIzaSyCAmlU3veZXfPlR2b65n-F1qjCPdG4EWH8"       # e.g., AIzaSyD... 
GOOGLE_CSE_ID  = "313c2d76cbd884814"     # e.g., 0123456789abcdefg:some-cx

def fetch_google_image_url(query: str) -> Optional[str]:
    """
    Tries a Google Custom Search for an image using query.
    If credentials are invalid or no result is found, returns None.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return None  # skip if no credentials

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "searchType": "image",
        "num": 1
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if items:
            return items[0].get("link")
        return None
    except Exception as e:
        logging.warning(f"Google CSE request failed: {e}")
        return None


# ------------------------------------------------------------------------------
# 2) Pydantic Models
# ------------------------------------------------------------------------------
class TeamModel(BaseModel):
    team_id: int
    name: str
    stage: str = Field(default="submit")

class GameModel(BaseModel):
    game_id: int
    team_id: int
    title: str
    image_url: Optional[str] = None
    embedding: Optional[List[float]] = None   # e.g. [0.23, 0.01, ...]

class VoteModel(BaseModel):
    vote_id: int
    team_id: int
    winner_id: int
    loser_id: int


# ------------------------------------------------------------------------------
# 3) Data Storage
# ------------------------------------------------------------------------------
DATA_DIR = "memory"
TEAMS_FILE = os.path.join(DATA_DIR, "teams.json")
GAMES_FILE = os.path.join(DATA_DIR, "games.json")
VOTES_FILE = os.path.join(DATA_DIR, "votes.json")
COUNTERS_FILE = os.path.join(DATA_DIR, "counters.json")

teams: Dict[int, TeamModel] = {}
games: Dict[int, GameModel] = {}
votes: Dict[int, VoteModel] = {}

team_counter = 1
game_counter = 1
vote_counter = 1


# ------------------------------------------------------------------------------
# 4) Load & Save
# ------------------------------------------------------------------------------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_data():
    global teams, games, votes
    global team_counter, game_counter, vote_counter

    if os.path.isfile(TEAMS_FILE):
        with open(TEAMS_FILE, "r") as f:
            raw = json.load(f)
            teams = {int(k): TeamModel(**v) for k, v in raw.items()}

    if os.path.isfile(GAMES_FILE):
        with open(GAMES_FILE, "r") as f:
            raw = json.load(f)
            games = {int(k): GameModel(**v) for k, v in raw.items()}

    if os.path.isfile(VOTES_FILE):
        with open(VOTES_FILE, "r") as f:
            raw = json.load(f)
            votes = {int(k): VoteModel(**v) for k, v in raw.items()}

    if os.path.isfile(COUNTERS_FILE):
        with open(COUNTERS_FILE, "r") as f:
            c = json.load(f)
            global team_counter, game_counter, vote_counter
            team_counter = c.get("team_counter", 1)
            game_counter = c.get("game_counter", 1)
            vote_counter = c.get("vote_counter", 1)

def save_data():
    with open(TEAMS_FILE, "w") as f:
        json.dump({k: v.model_dump() for k, v in teams.items()}, f, indent=2)
    with open(GAMES_FILE, "w") as f:
        json.dump({k: v.model_dump() for k, v in games.items()}, f, indent=2)
    with open(VOTES_FILE, "w") as f:
        json.dump({k: v.model_dump() for k, v in votes.items()}, f, indent=2)
    with open(COUNTERS_FILE, "w") as f:
        counters = {
            "team_counter": team_counter,
            "game_counter": game_counter,
            "vote_counter": vote_counter
        }
        json.dump(counters, f, indent=2)


# ------------------------------------------------------------------------------
# 5) CRUD
# ------------------------------------------------------------------------------
def create_team(name: str) -> int:
    global team_counter
    t_id = team_counter
    team_counter += 1
    teams[t_id] = TeamModel(team_id=t_id, name=name, stage="submit")
    save_data()
    return t_id

def remove_team(team_id: int):
    if team_id not in teams:
        return
    # remove games & votes
    to_delete_games = [gid for gid, g in games.items() if g.team_id == team_id]
    for gid in to_delete_games:
        del games[gid]

    to_delete_votes = [vid for vid, v in votes.items() if v.team_id == team_id]
    for vid in to_delete_votes:
        del votes[vid]

    del teams[team_id]
    save_data()

def set_team_stage(team_id: int, new_stage: str):
    if team_id in teams:
        t = teams[team_id]
        t.stage = new_stage
        teams[team_id] = t
        save_data()

def submit_game(team_id: int, title: str, image_url: Optional[str]) -> int:
    """
    If user didn't provide image_url, try google search. Otherwise fallback placeholder.
    Also generate random embedding for demonstration.
    """
    global game_counter
    g_id = game_counter
    game_counter += 1

    final_url = image_url
    if not final_url or final_url.strip() == "":
        found_url = fetch_google_image_url(title)
        if not found_url:
            # fallback
            found_url = f"https://via.placeholder.com/400?text={title.replace(' ', '+')}"
        final_url = found_url

    # random embedding
    embed_vec = list(np.random.rand(8).round(3))

    games[g_id] = GameModel(
        game_id=g_id,
        team_id=team_id,
        title=title,
        image_url=final_url,
        embedding=embed_vec
    )
    save_data()
    return g_id

def record_vote(team_id: int, winner_id: int, loser_id: int) -> int:
    global vote_counter
    v_id = vote_counter
    vote_counter += 1

    votes[v_id] = VoteModel(
        vote_id=v_id,
        team_id=team_id,
        winner_id=winner_id,
        loser_id=loser_id
    )
    save_data()
    return v_id

def get_games_for_team(team_id: int) -> List[GameModel]:
    return [g for g in games.values() if g.team_id == team_id]

def get_votes_for_team(team_id: int) -> List[VoteModel]:
    return [v for v in votes.values() if v.team_id == team_id]

def get_team_stage(team_id: int) -> str:
    if team_id in teams:
        return teams[team_id].stage
    return "none"


# ------------------------------------------------------------------------------
# 6) Helper: Distance-based Pair
# ------------------------------------------------------------------------------
def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    return float(np.sqrt(sum((a - b)**2 for a, b in zip(v1, v2))))

def find_max_distance_pair(games_list: List[GameModel]):
    max_dist = -1
    pair_out = None
    n = len(games_list)
    for i in range(n):
        for j in range(i + 1, n):
            emb_i = games_list[i].embedding or []
            emb_j = games_list[j].embedding or []
            dist = euclidean_distance(emb_i, emb_j)
            if dist > max_dist:
                max_dist = dist
                pair_out = (games_list[i], games_list[j])
    return pair_out


# ------------------------------------------------------------------------------
# 7) Bradley-Terry
# ------------------------------------------------------------------------------
def bradley_terry_analysis(df: pd.DataFrame, max_iters=1000, error_tol=1e-3) -> pd.Series:
    winsA = df.groupby('Excerpt A')['Wins A'].sum().reset_index()
    winsA = winsA[winsA['Wins A'] > 0]
    winsA.columns = ['Excerpt', 'Wins']

    winsB = df.groupby('Excerpt B')['Wins B'].sum().reset_index()
    winsB = winsB[winsB['Wins B'] > 0]
    winsB.columns = ['Excerpt', 'Wins']

    combined = pd.concat([winsA, winsB])
    wins = combined.groupby('Excerpt')['Wins'].sum()

    from collections import Counter
    num_games = Counter()
    for _, row in df.iterrows():
        pair_key = tuple(sorted([row['Excerpt A'], row['Excerpt B']]))
        total = row['Wins A'] + row['Wins B']
        num_games[pair_key] += total

    excerpts = sorted(set(df['Excerpt A']) | set(df['Excerpt B']))
    ranks = pd.Series(np.ones(len(excerpts)) / len(excerpts), index=excerpts)

    for iteration in range(max_iters):
        oldranks = ranks.copy()
        for ex in excerpts:
            ex_wins = wins.get(ex, 0.0)
            denom_sum = 0.0
            for p in excerpts:
                if p == ex:
                    continue
                pair = tuple(sorted([ex, p]))
                if pair not in num_games:
                    continue
                denom_sum += num_games[pair] / (ranks[ex] + ranks[p])
            ranks[ex] = ex_wins / denom_sum if denom_sum else 0.0

        total_ = ranks.sum()
        if total_ > 0:
            ranks /= total_

        if (ranks - oldranks).abs().sum() < error_tol:
            logging.info(f"Bradley-Terry converged after {iteration} iterations.")
            break

    # scale to [0..100]
    ranks = (ranks * 100).round(2)
    return ranks.sort_values(ascending=False)

def compute_bradley_terry_scores(team_id: int) -> Dict[int, float]:
    tvotes = get_votes_for_team(team_id)
    tgames = get_games_for_team(team_id)
    if len(tgames) < 2 or not tvotes:
        return {g.game_id: 0.0 for g in tgames}

    pair_agg = {}
    for v in tvotes:
        A = min(v.winner_id, v.loser_id)
        B = max(v.winner_id, v.loser_id)
        if (A, B) not in pair_agg:
            pair_agg[(A, B)] = [0, 0]
        if v.winner_id == A:
            pair_agg[(A, B)][0] += 1
        else:
            pair_agg[(A, B)][1] += 1

    rows = []
    for (A, B), (wa, wb) in pair_agg.items():
        rows.append({"Excerpt A": A, "Excerpt B": B, "Wins A": wa, "Wins B": wb})
    df = pd.DataFrame(rows, columns=["Excerpt A", "Excerpt B", "Wins A", "Wins B"])
    bt_series = bradley_terry_analysis(df)
    return {int(k): float(v) for k, v in bt_series.to_dict().items()}


# ------------------------------------------------------------------------------
# 8) Dash App
# ------------------------------------------------------------------------------
ensure_data_dir()
load_data()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Game Preference (Unified Callback)"
app.config.suppress_callback_exceptions = True

def get_team_options():
    if not teams:
        return []
    return [{"label": f"{t.team_id} - {t.name}", "value": t.team_id} for t in teams.values()]

app.layout = dbc.Container([
    html.H1("Game Preference (Bradley-Terry + Google Image Search)", className="mt-3"),

    # Create/Remove Teams in a single callback
    dbc.Row([
        dbc.Col([
            html.H5("Create or Remove Team"),
            dbc.Input(id="create-team-name", placeholder="Team name...", n_submit=0),
            dbc.Button("Create Team", id="btn-create-team", color="primary", className="mt-2 me-2"),
            dbc.Button("Remove Selected Team", id="btn-remove-team", color="danger", className="mt-2"),
            html.Div(id="create-team-output", className="mt-2 text-success"),
            html.Div(id="remove-team-output", className="text-danger")
        ], width=4),

        dbc.Col([
            html.H5("Existing Teams"),
            dcc.Dropdown(
                id="team-select",
                options=get_team_options(),
                placeholder="Select a team...",
                value=None
            ),
            html.Div(id="team-stage-display", className="mt-2"),
        ], width=4),
    ], className="mt-4"),

    html.Hr(),

    # Admin stage
    html.H5("Admin Controls (Change Stage)"),
    dbc.RadioItems(
        id="radio-stage",
        options=[
            {"label": "Submit",  "value": "submit"},
            {"label": "Vote",    "value": "vote"},
            {"label": "Results", "value": "results"}
        ],
        value="submit",
        inline=True
    ),
    dbc.Button("Update Stage", id="btn-update-stage", color="secondary", className="ms-2"),

    html.Hr(),

    # Dynamic content
    html.Div(id="stage-content"),
], fluid=True)

# ------------------------------------------------------------------------------
# 9) Unified Callback for Creating & Removing Teams
# ------------------------------------------------------------------------------
@app.callback(
    Output("team-select", "options"),
    Output("create-team-output", "children"),
    Output("remove-team-output", "children"),
    Input("btn-create-team", "n_clicks"),
    Input("btn-remove-team", "n_clicks"),
    State("create-team-name", "value"),
    State("team-select", "value"),
    prevent_initial_call=True
)
def create_or_remove_team(n_create, n_remove, new_team_name, remove_team_id):
    """
    We unify the creation and removal of teams into a single callback,
    avoiding "Output is already in use" duplicates.
    - If "Create Team" triggered, we create a new team.
    - If "Remove Team" triggered, we remove the currently selected team.
    - Both actions update the same 'team-select.options'.
    """
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, "", ""

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "btn-create-team":
        # Create logic
        if not new_team_name:
            return get_team_options(), "Please enter a team name.", ""
        t_id = create_team(new_team_name)
        msg = f"Team '{new_team_name}' created (ID={t_id})."
        return get_team_options(), msg, ""

    elif triggered_id == "btn-remove-team":
        # Remove logic
        if not remove_team_id:
            return get_team_options(), "", "No team selected to remove."
        remove_team(remove_team_id)
        return get_team_options(), "", f"Team {remove_team_id} removed."

    # default fallback
    return get_team_options(), "", ""

# ------------------------------------------------------------------------------
# 10) Stage Display
# ------------------------------------------------------------------------------
@app.callback(
    Output("team-stage-display", "children"),
    Output("radio-stage", "value"),
    Input("team-select", "value")
)
def update_stage_display(team_id):
    if not team_id:
        return ("No team selected.", "submit")
    stg = get_team_stage(team_id)
    return (f"Current Stage: {stg}", stg)

@app.callback(
    Output("team-stage-display", "children", allow_duplicate=True),
    Input("btn-update-stage", "n_clicks"),
    State("team-select", "value"),
    State("radio-stage", "value"),
    prevent_initial_call=True
)
def admin_set_stage_cb(n_clicks, team_id, new_stage):
    if not team_id:
        return "No team selected."
    set_team_stage(team_id, new_stage)
    return f"Current Stage: {new_stage}"

# ------------------------------------------------------------------------------
# 11) Render Stage-Specific Content
# ------------------------------------------------------------------------------
@app.callback(
    Output("stage-content", "children"),
    Input("team-select", "value"),
    Input("radio-stage", "value")
)
def render_stage_content(team_id, stage):
    if not team_id:
        return html.Div("Select or create a team first.")
    if stage == "submit":
        return render_submit_stage(team_id)
    elif stage == "vote":
        return render_vote_stage(team_id)
    elif stage == "results":
        return render_results_stage(team_id)
    else:
        return html.Div("Unknown stage.")

def render_submit_stage(team_id: int):
    return html.Div([
        html.H4("Submit New Game"),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="input-game-title", placeholder="Game Title", n_submit=0),
                dbc.Input(id="input-game-image", placeholder="Image URL (optional)",
                          className="mt-2", n_submit=0),
                dbc.Button("Submit Game", id="btn-submit-game", color="primary", className="mt-2"),
                html.Div(id="submit-game-output", className="mt-2 text-success")
            ], width=4),

            dbc.Col([
                html.H6("Existing Games in This Team:"),
                html.Div(id="submit-game-list")
            ], width=8)
        ])
    ])

# Single callback for listing & submitting
@app.callback(
    Output("submit-game-output", "children"),
    Output("submit-game-list", "children"),
    Output("input-game-title", "value"),
    Output("input-game-image", "value"),
    Input("btn-submit-game", "n_clicks"),
    Input("team-select", "value"),
    Input("input-game-title", "n_submit"),
    Input("input-game-image", "n_submit"),
    State("input-game-title", "value"),
    State("input-game-image", "value"),
    prevent_initial_call=True
)
def handle_submit_game_or_update_list(
    btn_clicks, team_id, title_n_submit, image_n_submit, title, image_url
):
    if not team_id:
        return ("No team selected.", "No games yet.", "", "")

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "team-select":
        # user just switched teams -> refresh list only
        return ("", render_game_list(team_id), "", "")

    # user triggered a submission by button or pressing Enter
    if not title:
        return ("No title provided.", render_game_list(team_id), "", "")

    g_id = submit_game(team_id, title, image_url)
    msg = f"Game '{title}' submitted (ID={g_id})."
    return (msg, render_game_list(team_id), "", "")

def render_game_list(team_id: int):
    team_gs = get_games_for_team(team_id)
    if not team_gs:
        return "No games submitted yet."
    items = []
    for g in team_gs:
        items.append(html.Li(f"[ID={g.game_id}] {g.title} - {g.image_url or 'No Image'}"))
    return html.Ul(items)

# ------------------------------------------------------------------------------
# 12) Vote Stage
# ------------------------------------------------------------------------------
def render_vote_stage(team_id: int):
    if len(get_games_for_team(team_id)) < 2:
        return html.Div("Not enough games to vote on.")

    return html.Div([
        html.H4("Pairwise Voting"),
        html.Div([
            dbc.RadioItems(
                id="radio-vote-method",
                options=[
                    {"label": "Random Pair", "value": "random"},
                    {"label": "Max Distance Pair", "value": "distance"}
                ],
                value="random",
                inline=True
            ),
            dbc.Button("Next Pair", id="btn-next-pair", color="primary", className="ms-2"),
        ]),
        html.Div(id="vote-pair-container", className="mt-3"),
        html.Div(id="vote-output", className="mt-2 text-success"),
    ])

@app.callback(
    Output("vote-pair-container", "children"),
    Input("btn-next-pair", "n_clicks"),
    State("team-select", "value"),
    State("radio-vote-method", "value")
)
def display_pair(n_clicks, team_id, vote_method):
    gs = get_games_for_team(team_id)
    if len(gs) < 2:
        return html.Div("Not enough games to vote on.")

    if vote_method == "distance":
        pair = find_max_distance_pair(gs)
    else:
        # random
        pair = random.sample(gs, 2)

    if not pair:
        return html.Div("Could not find a pair.")
    gA, gB = pair
    style = {"border": "1px solid #ccc", "padding": "10px", "borderRadius": "5px", "margin": "5px"}

    return dbc.Row([
        dbc.Col([
            html.H5(gA.title),
            html.Img(src=gA.image_url, style={"width": "100%", "height": "auto"}) if gA.image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": gA.game_id},
                       color="success", className="mt-2")
        ], width=6, style=style),

        dbc.Col([
            html.H5(gB.title),
            html.Img(src=gB.image_url, style={"width": "100%", "height": "auto"}) if gB.image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": gB.game_id},
                       color="success", className="mt-2")
        ], width=6, style=style),
    ])

@app.callback(
    Output("vote-output", "children"),
    [Input({"type": "btn-vote-winner", "index": dash.ALL}, "n_clicks")],
    [State({"type": "btn-vote-winner", "index": dash.ALL}, "id"),
     State("team-select", "value")],
    prevent_initial_call=True
)
def handle_vote(n_clicks_list, button_ids, team_id):
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered_idx = None
    for i, val in enumerate(n_clicks_list):
        if val and val > 0:
            triggered_idx = i
            break
    if triggered_idx is None:
        raise dash.exceptions.PreventUpdate

    winner_game_id = button_ids[triggered_idx]["index"]
    all_game_ids = [b["index"] for b in button_ids]
    loser_candidates = [x for x in all_game_ids if x != winner_game_id]
    if not loser_candidates:
        return "Error: No loser found."

    loser_game_id = loser_candidates[0]
    record_vote(team_id, winner_game_id, loser_game_id)
    return f"Vote Recorded! Winner={winner_game_id}, Loser={loser_game_id}."


# ------------------------------------------------------------------------------
# 13) Results Stage
# ------------------------------------------------------------------------------
def render_results_stage(team_id: int):
    scores = compute_bradley_terry_scores(team_id)
    if not scores:
        return html.Div("No votes or insufficient data to compute results.")

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = []
    rank_count = 1
    for gid, sc in sorted_items:
        title = games[gid].title
        rows.append(html.Tr([
            html.Td(rank_count),
            html.Td(title),
            html.Td(f"{sc:.2f}")
        ]))
        rank_count += 1

    fig = px.bar(
        x=[games[g].title for g, s in sorted_items],
        y=[s for g, s in sorted_items],
        labels={"x": "Game", "y": "Score"},
        title="Bradley-Terry Scores"
    )
    fig.update_layout(height=500)

    return html.Div([
        html.H4("Results"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Rank"), html.Th("Game"), html.Th("Score")]))] +
            [html.Tbody(rows)],
            bordered=True
        ),
        dcc.Graph(figure=fig, style={"width": "100%"}),
        html.Div("Higher score indicates stronger preference from pairwise votes.")
    ])


# ------------------------------------------------------------------------------
# 14) Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run_server(debug=True)