"""
Full Dash App with Bradley-Terry Iterative Algorithm & Local JSON Persistence
-----------------------------------------------------------------------------
- Automatically loads existing teams (and other data) on startup, showing them
  in the team dropdown.
- Uses Pydantic models (compatible with Pydantic 2.x by calling model_dump()
  instead of .dict()).
- Stores data (teams, games, votes) in local JSON files in a 'memory' folder
  for persistence between app runs.
- Implements an iterative Bradley-Terry approach to compute preference scores
  from pairwise votes (A/B). Accounts for zero-win games to avoid KeyErrors.
- Uses suppress_callback_exceptions=True to allow dynamic (stage-based) layout
  rendering.

How to Run:
-----------
1) Ensure you have a folder named 'memory' in the same directory as this script,
   or let the script create it automatically.
2) pip install dash dash_bootstrap_components plotly pydantic pandas numpy
3) python app.py
4) Open http://127.0.0.1:8050 in your browser.

Flow:
-----
- Create a Team (or select an existing one if present in the dropdown).
- Stage = "Submit": add games.
- Stage = "Vote": do pairwise votes.
- Stage = "Results": see final ranking using Bradley-Terry.

Note:
-----
If you see a warning or error about aggregator usage in pandas, we switched
to direct `.sum()` calls so it should be fine. For Pydantic 2, we replaced
`.dict()` with `.model_dump()`.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from collections import Counter
import random

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1) Pydantic Models (Use model_dump() for Pydantic v2+)
# ------------------------------------------------------------------------------
class TeamModel(BaseModel):
    team_id: int
    name: str
    stage: str = Field(default="submit")  # "submit", "vote", or "results"

class GameModel(BaseModel):
    game_id: int
    team_id: int
    title: str
    image_url: Optional[str] = None

class VoteModel(BaseModel):
    vote_id: int
    team_id: int
    winner_id: int
    loser_id: int


# ------------------------------------------------------------------------------
# 2) File Storage Paths
# ------------------------------------------------------------------------------
DATA_DIR = "memory"
TEAMS_FILE = os.path.join(DATA_DIR, "teams.json")
GAMES_FILE = os.path.join(DATA_DIR, "games.json")
VOTES_FILE = os.path.join(DATA_DIR, "votes.json")
COUNTERS_FILE = os.path.join(DATA_DIR, "counters.json")


# ------------------------------------------------------------------------------
# 3) In-Memory Data Structures
# ------------------------------------------------------------------------------
teams: Dict[int, TeamModel] = {}
games: Dict[int, GameModel] = {}
votes: Dict[int, VoteModel] = {}

team_counter = 1
game_counter = 1
vote_counter = 1


# ------------------------------------------------------------------------------
# 4) Data Persistence: Load & Save
# ------------------------------------------------------------------------------
def ensure_data_dir():
    """Create the 'memory' folder if missing."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_data():
    """Load existing data from JSON into in-memory structures."""
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
            counters = json.load(f)
            team_counter = counters.get("team_counter", 1)
            game_counter = counters.get("game_counter", 1)
            vote_counter = counters.get("vote_counter", 1)

def save_data():
    """Save in-memory data to local JSON files."""
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
# 5) CRUD Functions
# ------------------------------------------------------------------------------
def create_team(name: str) -> int:
    global team_counter
    t_id = team_counter
    team_counter += 1

    teams[t_id] = TeamModel(team_id=t_id, name=name, stage="submit")
    save_data()
    return t_id

def set_team_stage(team_id: int, new_stage: str):
    if team_id in teams:
        t = teams[team_id]
        t.stage = new_stage
        teams[team_id] = t
        save_data()

def submit_game(team_id: int, title: str, image_url: Optional[str]) -> int:
    global game_counter
    g_id = game_counter
    game_counter += 1

    games[g_id] = GameModel(
        game_id=g_id,
        team_id=team_id,
        title=title,
        image_url=image_url
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
# 6) Bradley-Terry Iterative Implementation
# ------------------------------------------------------------------------------
def bradley_terry_analysis(df: pd.DataFrame,
                           max_iters: int = 1000,
                           error_tol: float = 1e-3) -> pd.Series:
    """
    Iterative Bradley-Terry approach to compute preference scores from pairwise data.
    df must have columns: ['Excerpt A', 'Excerpt B', 'Wins A', 'Wins B'].

    Returns a pd.Series indexed by excerpt (game_id), sorted descending,
    with values scaled ~ [0..100].
    """
    # Summation of wins for A and B
    # Use .sum() to avoid aggregator warnings
    winsA = df.groupby('Excerpt A')['Wins A'].sum().reset_index()
    winsA = winsA[winsA['Wins A'] > 0]
    winsA.columns = ['Excerpt', 'Wins']

    winsB = df.groupby('Excerpt B')['Wins B'].sum().reset_index()
    winsB = winsB[winsB['Wins B'] > 0]
    winsB.columns = ['Excerpt', 'Wins']

    # Combine total wins per excerpt
    wins = pd.concat([winsA, winsB]).groupby('Excerpt')['Wins'].sum()

    # Count total matchups
    num_games = Counter()
    for _, row in df.iterrows():
        pair_key = tuple(sorted([row['Excerpt A'], row['Excerpt B']]))
        total = row['Wins A'] + row['Wins B']
        num_games[pair_key] += total

    # All excerpt IDs
    excerpts = sorted(set(df['Excerpt A']) | set(df['Excerpt B']))

    # Initialize ranks
    ranks = pd.Series(np.ones(len(excerpts)) / len(excerpts), index=excerpts)

    # Iteration
    for iteration in range(max_iters):
        oldranks = ranks.copy()

        for ex in excerpts:
            ex_wins = wins.get(ex, 0.0)  # If ex never had wins, default 0
            denom_sum = 0.0
            for p in excerpts:
                if p == ex:
                    continue
                pair = tuple(sorted([ex, p]))
                # if they never faced each other, skip
                if pair not in num_games:
                    continue
                # standard BT denominator
                denom_sum += num_games[pair] / (ranks[ex] + ranks[p])

            # update rank
            if denom_sum != 0:
                ranks[ex] = ex_wins / denom_sum
            else:
                ranks[ex] = 0.0

        # normalize
        total_ranks = ranks.sum()
        if total_ranks > 0:
            ranks /= total_ranks

        # check convergence
        diff = (ranks - oldranks).abs().sum()
        if diff < error_tol:
            logging.info(f" * Converged after {iteration} iterations.")
            break

    # scale to [0..100] for readability
    ranks = (ranks * 100).round(2)
    return ranks.sort_values(ascending=False)

def compute_bradley_terry_scores(team_id: int) -> Dict[int, float]:
    """
    Converts team votes into a DataFrame for bradley_terry_analysis.
    Returns a dict: { game_id: score (0..100) }.
    """
    team_votes = get_votes_for_team(team_id)
    team_games = get_games_for_team(team_id)

    # If no data or only 1 game, no ranking possible
    if len(team_games) < 2 or not team_votes:
        return {g.game_id: 0.0 for g in team_games}

    # Build DataFrame with columns: Excerpt A, Excerpt B, Wins A, Wins B
    pair_agg = {}
    for v in team_votes:
        A = min(v.winner_id, v.loser_id)
        B = max(v.winner_id, v.loser_id)
        if (A, B) not in pair_agg:
            pair_agg[(A, B)] = [0, 0]  # [winsA, winsB]
        # increment the winner's position
        if v.winner_id == A:
            pair_agg[(A, B)][0] += 1
        else:
            pair_agg[(A, B)][1] += 1

    rows = []
    for (A, B), (winsA, winsB) in pair_agg.items():
        rows.append({
            'Excerpt A': A,
            'Excerpt B': B,
            'Wins A': winsA,
            'Wins B': winsB
        })
    df = pd.DataFrame(rows, columns=['Excerpt A', 'Excerpt B', 'Wins A', 'Wins B'])

    bt_series = bradley_terry_analysis(df, max_iters=1000, error_tol=1e-3)
    results = bt_series.to_dict()   # e.g. { 2: 75.4, 3: 24.6 }
    results = {int(k): float(v) for k, v in results.items()}
    return results


# ------------------------------------------------------------------------------
# 7) Dash App
# ------------------------------------------------------------------------------
# Load existing data first, so the initial layout can reflect existing teams
ensure_data_dir()
load_data()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
app.title = "Game Preference (Bradley-Terry)"

def get_team_options():
    """Generate a list of 'label/value' pairs for the team dropdown."""
    if not teams:
        return []
    return [{"label": f"{t.team_id} - {t.name}", "value": t.team_id} for t in teams.values()]

app.layout = dbc.Container([
    html.H1("Game Preference (Bradley-Terry)"),

    # Row: Create/Select Team
    dbc.Row([
        dbc.Col([
            html.H5("Create a New Team"),
            dbc.Input(id="create-team-name", placeholder="Team name..."),
            dbc.Button("Create Team", id="btn-create-team", color="primary", className="mt-2"),
            html.Div(id="create-team-output", className="mt-2 text-success")
        ], width=4),

        dbc.Col([
            html.H5("Existing Teams"),
            dcc.Dropdown(
                id="team-select",
                options=get_team_options(),   # Pre-populate with existing teams
                placeholder="Select a team...",
                value=None
            ),
            html.Div(id="team-stage-display", className="mt-2"),
        ], width=4),
    ], className="mt-4"),

    html.Hr(),

    # Admin stage controls
    html.H5("Admin Controls (Change Stage)"),
    dbc.RadioItems(
        id="radio-stage",
        options=[
            {"label": "Submit",  "value": "submit"},
            {"label": "Vote",    "value": "vote"},
            {"label": "Results", "value": "results"}
        ],
        value="submit", inline=True
    ),
    dbc.Button("Update Stage", id="btn-update-stage", color="secondary", className="ms-2"),

    html.Hr(),

    # Dynamic content area
    html.Div(id="stage-content"),

], fluid=True)


# ------------------------------------------------------------------------------
# 8) Callbacks
# ------------------------------------------------------------------------------
@app.callback(
    Output("create-team-output", "children"),
    Output("team-select", "options"),
    Input("btn-create-team", "n_clicks"),
    State("create-team-name", "value"),
    prevent_initial_call=True
)
def create_team_callback(n_clicks, new_team_name):
    if not new_team_name:
        return ("Please enter a team name.", dash.no_update)
    t_id = create_team(new_team_name)
    msg = f"Team '{new_team_name}' created (ID={t_id})."
    # Refresh the dropdown options
    opts = get_team_options()
    return (msg, opts)

@app.callback(
    Output("team-stage-display", "children"),
    Output("radio-stage", "value"),
    Input("team-select", "value")
)
def update_stage_display(team_id):
    """When the user selects a team, show its current stage."""
    if not team_id:
        return ("No team selected.", "submit")
    current_stage = get_team_stage(team_id)
    return (f"Current Stage: {current_stage}", current_stage)

@app.callback(
    Output("team-stage-display", "children", allow_duplicate=True),
    Input("btn-update-stage", "n_clicks"),
    State("team-select", "value"),
    State("radio-stage", "value"),
    prevent_initial_call=True
)
def admin_set_stage_cb(n_clicks, team_id, new_stage):
    """Admin sets the stage for the selected team."""
    if not team_id:
        return "No team selected."
    set_team_stage(team_id, new_stage)
    return f"Current Stage: {new_stage}"

@app.callback(
    Output("stage-content", "children"),
    Input("team-select", "value"),
    Input("radio-stage", "value")
)
def render_stage_content(team_id, stage):
    """Render dynamic content depending on the selected team's stage."""
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

# ------------------------------------------------------------------------------
# 9) Stage: Submit
# ------------------------------------------------------------------------------
def render_submit_stage(team_id: int):
    return html.Div([
        html.H4("Submit New Game"),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="input-game-title", placeholder="Game Title"),
                dbc.Input(id="input-game-image", placeholder="Image URL (optional)", className="mt-2"),
                dbc.Button("Submit Game", id="btn-submit-game", color="primary", className="mt-2"),
                html.Div(id="submit-game-output", className="mt-2 text-success")
            ], width=4),

            dbc.Col([
                html.H6("Existing Games in This Team:"),
                html.Div(id="submit-game-list")
            ], width=8)
        ])
    ])

@app.callback(
    Output("submit-game-output", "children"),
    Output("submit-game-list", "children"),
    Input("btn-submit-game", "n_clicks"),
    Input("team-select", "value"),
    State("input-game-title", "value"),
    State("input-game-image", "value"),
    prevent_initial_call=True
)
def handle_submit_game_or_update_list(n_clicks, team_id, title, image_url):
    """Submits a new game if the button was clicked, or updates the list if user switched teams."""
    if not team_id:
        return ("No team selected.", "No games yet.")

    ctx = callback_context
    if not ctx.triggered:
        return ("", "No games yet.")

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If user just switched teams, only update the list
    if triggered_id == "team-select":
        return ("", render_game_list(team_id))

    # Otherwise user submitted a new game
    if not title:
        return ("No title provided.", render_game_list(team_id))

    g_id = submit_game(team_id, title, image_url)
    msg = f"Game '{title}' submitted (ID={g_id})."
    return (msg, render_game_list(team_id))

def render_game_list(team_id: int):
    team_gs = get_games_for_team(team_id)
    if not team_gs:
        return "No games submitted yet."
    items = []
    for g in team_gs:
        items.append(html.Li(f"[ID={g.game_id}] {g.title} - {g.image_url or 'No Image'}"))
    return html.Ul(items)

# ------------------------------------------------------------------------------
# 10) Stage: Vote
# ------------------------------------------------------------------------------
def render_vote_stage(team_id: int):
    """Renders the voting UI if there are at least 2 games."""
    if len(get_games_for_team(team_id)) < 2:
        return html.Div("Not enough games to vote on.")

    return html.Div([
        html.H4("Pairwise Voting"),
        html.Div(id="vote-pair-container"),
        dbc.Button("Next Pair", id="btn-next-pair", color="primary", className="mt-3"),
        html.Div(id="vote-output", className="mt-2 text-success"),
    ])

@app.callback(
    Output("vote-pair-container", "children"),
    Input("btn-next-pair", "n_clicks"),
    State("team-select", "value")
)
def display_random_pair(n_clicks, team_id):
    all_games = get_games_for_team(team_id)
    if len(all_games) < 2:
        return html.Div("Not enough games to vote on.")

    pair = random.sample(all_games, 2)
    style = {"border": "1px solid #ccc", "padding": "10px", "borderRadius": "5px", "margin": "5px"}

    return dbc.Row([
        dbc.Col([
            html.H5(pair[0].title),
            html.Img(src=pair[0].image_url, style={"width": "100%", "height": "auto"}) if pair[0].image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": pair[0].game_id},
                       color="success", className="mt-2")
        ], width=6, style=style),

        dbc.Col([
            html.H5(pair[1].title),
            html.Img(src=pair[1].image_url, style={"width": "100%", "height": "auto"}) if pair[1].image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": pair[1].game_id},
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
    """Records the user's vote for whichever 'Choose This' button they clicked."""
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered_idx = None
    for i, click_val in enumerate(n_clicks_list):
        if click_val and click_val > 0:
            triggered_idx = i
            break
    if triggered_idx is None:
        raise dash.exceptions.PreventUpdate

    winner_game_id = button_ids[triggered_idx]["index"]
    # The other is the loser
    all_game_ids = [b["index"] for b in button_ids]
    loser_candidates = [x for x in all_game_ids if x != winner_game_id]
    if not loser_candidates:
        return "Error: Could not determine loser."

    loser_game_id = loser_candidates[0]
    record_vote(team_id, winner_game_id, loser_game_id)
    return f"Vote Recorded! Winner={winner_game_id}, Loser={loser_game_id}."

# ------------------------------------------------------------------------------
# 11) Stage: Results
# ------------------------------------------------------------------------------
def render_results_stage(team_id: int):
    """Shows final Bradley-Terry ranking and bar chart."""
    scores = compute_bradley_terry_scores(team_id)
    if not scores:
        return html.Div("No votes or insufficient data to compute results.")

    # Sort desc by score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = []
    rank_count = 1
    for g_id, scr in sorted_items:
        title = games[g_id].title
        rows.append(html.Tr([
            html.Td(rank_count),
            html.Td(title),
            html.Td(f"{scr:.2f}")
        ]))
        rank_count += 1

    fig = px.bar(
        x=[games[g_id].title for g_id, s in sorted_items],
        y=[s for g_id, s in sorted_items],
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
# 12) Main Entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run_server(debug=True)
