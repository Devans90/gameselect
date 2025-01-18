"""
Example Implementation Using Dash (Plotly) + Pydantic Models (In-Memory Storage)
-------------------------------------------------------------------------------
This is a minimal prototype illustrating how you could build a self-hosted
app for:
  1. Submitting games (with optional images) for a specific Team.
  2. Pairwise (A/B) voting on those games.
  3. Generating results using a simple Bradley-Terry-like approach.

Key Points:
 - In-memory storage is used for simplicity. In production, use a database.
 - We show a naive logistic regression approach (via scikit-learn) to approximate
   Bradley-Terry; you can swap in a dedicated library (e.g. pybradleyterry2) if you prefer.
 - This code can be run directly with `python <filename>.py` once Dash and required
   dependencies are installed. Then open http://127.0.0.1:8050 in your browser.
 - Minimal error handling and no authentication. Meant only as a starting prototype.

Dependencies (Install Separately):
 - dash, dash_bootstrap_components
 - pydantic
 - scikit-learn
 - plotly
"""

from typing import Optional, List, Dict
from pydantic import BaseModel
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np

from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------------------
# 1) Pydantic Models
# -------------------------------------------------------------------------------------

class TeamModel(BaseModel):
    team_id: int
    name: str
    stage: str  # one of ["submit", "vote", "results"]

class GameModel(BaseModel):
    game_id: int
    team_id: int
    title: str
    image_url: Optional[str] = None

class VoteModel(BaseModel):
    """
    Represents a single pairwise comparison result (A vs. B).
    winner_id: The ID of the game that won.
    loser_id:  The ID of the game that lost.
    """
    vote_id: int
    team_id: int
    winner_id: int
    loser_id: int

class UserModel(BaseModel):
    user_id: int
    username: str

# -------------------------------------------------------------------------------------
# 2) In-Memory Data Store (Dictionaries for simplicity)
# -------------------------------------------------------------------------------------
teams: Dict[int, TeamModel] = {}
games: Dict[int, GameModel] = {}
votes: Dict[int, VoteModel] = {}
users: Dict[int, UserModel] = {}

# We'll keep counters to auto-increment IDs
team_counter = 1
game_counter = 1
vote_counter = 1
user_counter = 1

# -------------------------------------------------------------------------------------
# 3) Helper / CRUD Functions
# -------------------------------------------------------------------------------------

def create_team(name: str) -> int:
    """
    Creates a new Team in 'submit' stage and returns the team_id.
    """
    global team_counter
    t_id = team_counter
    team_counter += 1

    new_team = TeamModel(team_id=t_id, name=name, stage="submit")
    teams[t_id] = new_team
    return t_id

def set_team_stage(team_id: int, new_stage: str) -> None:
    """
    Updates the stage of a team (e.g. from 'submit' to 'vote' to 'results').
    """
    if team_id in teams:
        team = teams[team_id]
        team.stage = new_stage
        teams[team_id] = team

def submit_game(team_id: int, title: str, image_url: Optional[str] = None) -> int:
    """
    Add a new game to the specified team.
    """
    global game_counter
    g_id = game_counter
    game_counter += 1

    new_game = GameModel(
        game_id=g_id,
        team_id=team_id,
        title=title,
        image_url=image_url
    )
    games[g_id] = new_game
    return g_id

def record_vote(team_id: int, winner_id: int, loser_id: int) -> int:
    """
    Stores a new vote (winner over loser) for the given team.
    """
    global vote_counter
    v_id = vote_counter
    vote_counter += 1

    new_vote = VoteModel(
        vote_id=v_id,
        team_id=team_id,
        winner_id=winner_id,
        loser_id=loser_id
    )
    votes[v_id] = new_vote
    return v_id

def get_games_for_team(team_id: int) -> List[GameModel]:
    """
    Returns a list of all games belonging to the specified team.
    """
    return [g for g in games.values() if g.team_id == team_id]

def get_votes_for_team(team_id: int) -> List[VoteModel]:
    """
    Returns a list of all votes for the specified team.
    """
    return [v for v in votes.values() if v.team_id == team_id]

def get_team_stage(team_id: int) -> str:
    """
    Returns the current stage of the team, or 'none' if not found.
    """
    if team_id in teams:
        return teams[team_id].stage
    return "none"

# -------------------------------------------------------------------------------------
# 4) Bradley-Terry Approx (via Logistic Regression)
# -------------------------------------------------------------------------------------

def compute_bradley_terry_scores(team_id: int) -> Dict[int, float]:
    """
    Approximates Bradley-Terry scores using logistic regression.
    Each game is turned into a one-hot vector, and the pairwise 
    preference is the training signal.

    Returns:
      A dict { game_id: estimated_score }
    """
    team_votes = get_votes_for_team(team_id)
    team_games = get_games_for_team(team_id)
    if not team_votes or len(team_games) < 2:
        return {g.game_id: 0.0 for g in team_games}

    # Step 1: Map game_ids to indices
    game_id_to_idx = {g.game_id: i for i, g in enumerate(team_games)}
    n_games = len(team_games)

    # Step 2: Build training data
    # We will create a feature vector for each pair:
    # One-hot with shape (n_games), +1 for winner, -1 for loser
    # Label = 1 for winner-loser, 0 for loser-winner in logistic regression
    X = []
    y = []
    for v in team_votes:
        row = np.zeros(n_games)
        win_idx = game_id_to_idx[v.winner_id]
        lose_idx = game_id_to_idx[v.loser_id]
        # We'll do +1 for the winner, -1 for the loser
        row[win_idx] = 1
        row[lose_idx] = -1
        X.append(row)
        # The target is 1, meaning "winner got picked over loser"
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Fit logistic regression
    # By default, scikit-learn logistic regression expects a binary classification
    # We are basically modeling sign(X * beta) = y. We'll do a standard approach.
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # The coefficients reflect the relative skill for each game
    # The higher the coefficient, the stronger the preference.
    # We can interpret them as the 'beta' in Bradley-Terry-like log-odds.
    coefs = model.coef_.flatten()  # shape (n_games,)

    # We'll store them in a dict
    game_scores = {}
    for g, idx in game_id_to_idx.items():
        game_scores[g] = coefs[idx]

    return game_scores

# -------------------------------------------------------------------------------------
# 5) Dash App
# -------------------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Game Preference App (Bradley-Terry)"),

    # Team Creation / Selection
    dbc.Row([
        dbc.Col([
            html.H5("Create a New Team"),
            dbc.Input(id="create-team-name", placeholder="Team name..."),
            dbc.Button("Create Team", id="btn-create-team", color="primary", className="mt-2"),
            html.Div(id="create-team-output", className="mt-1 text-success")
        ], width=4),

        dbc.Col([
            html.H5("Existing Teams"),
            dcc.Dropdown(
                id="team-select",
                options=[],
                placeholder="Select a team...",
                value=None
            ),
            html.Div(id="team-stage-display", className="mt-2"),
        ], width=4),
    ], className="mt-4"),

    html.Hr(),

    # Admin Stage Controls
    html.H5("Admin Controls (Change Stage)"),
    dbc.RadioItems(
        id="radio-stage",
        options=[
            {"label": "Submit", "value": "submit"},
            {"label": "Vote", "value": "vote"},
            {"label": "Results", "value": "results"}
        ],
        value="submit",
        inline=True
    ),
    dbc.Button("Update Stage", id="btn-update-stage", color="secondary", className="ms-2"),

    html.Hr(),

    # CONTENT AREA: Will change based on stage
    html.Div(id="stage-content"),

], fluid=True)

# -------------------------------------------------------------------------------------
# 6) Callbacks
# -------------------------------------------------------------------------------------

@app.callback(
    Output("create-team-output", "children"),
    Output("team-select", "options"),
    Input("btn-create-team", "n_clicks"),
    State("create-team-name", "value"),
    prevent_initial_call=True
)
def create_team_callback(n_clicks: int, name: str):
    """
    Creates a new team, updates the dropdown.
    """
    if not name:
        return ("No team name provided.", dash.no_update)
    t_id = create_team(name)
    msg = f"Team '{name}' created with ID={t_id}."

    # Update dropdown options
    opts = [{"label": f"{t.team_id} - {t.name}", "value": t.team_id} for t in teams.values()]
    return (msg, opts)

@app.callback(
    Output("team-stage-display", "children"),
    Output("radio-stage", "value"),
    Input("team-select", "value")
)
def display_team_stage(team_id: int):
    """
    Shows the current stage of the selected team and sets the stage radio to match.
    """
    if not team_id:
        return ("No team selected.", "submit")

    current_stage = get_team_stage(team_id)
    return (f"Current Stage: {current_stage}", current_stage)

@app.callback(
    Output("stage-content", "children"),
    Input("team-select", "value"),
    Input("radio-stage", "value")
)
def render_stage_content(team_id: int, stage: str):
    """
    Renders the appropriate UI for the current stage of the selected team.
    """
    if not team_id:
        return html.Div("Select or create a team first.")

    if stage == "submit":
        return render_submit_stage(team_id)
    elif stage == "vote":
        return render_vote_stage(team_id)
    elif stage == "results":
        return render_results_stage(team_id)
    else:
        return html.Div("Unknown stage")

@app.callback(
    Output("team-stage-display", "children", allow_duplicate=True),
    Input("btn-update-stage", "n_clicks"),
    State("team-select", "value"),
    State("radio-stage", "value"),
    prevent_initial_call=True
)
def admin_set_stage(n_clicks: int, team_id: int, stage: str):
    """
    Allows admin to change the stage of the selected team.
    """
    if not team_id:
        return "No team selected."
    set_team_stage(team_id, stage)
    return f"Current Stage: {stage}"

# -------------------------------------------------------------------------------------
# 7) Stage-Specific Render Functions
# -------------------------------------------------------------------------------------

def render_submit_stage(team_id: int):
    """
    Returns layout allowing the user to submit new games.
    """
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
    State("team-select", "value"),
    State("input-game-title", "value"),
    State("input-game-image", "value"),
    prevent_initial_call=True
)
def handle_submit_game(n_clicks: int, team_id: int, title: str, image_url: str):
    if not title:
        return ("No title provided.", dash.no_update)
    g_id = submit_game(team_id, title, image_url)
    msg = f"Game '{title}' submitted (ID={g_id})."

    # Show updated list
    return (msg, render_game_list_html(team_id))

@app.callback(
    Output("submit-game-list", "children"),
    Input("team-select", "value"),
    prevent_initial_call=True
)
def update_submit_game_list(team_id: int):
    """
    Updates the list of submitted games whenever a new team is selected.
    """
    return render_game_list_html(team_id)

def render_game_list_html(team_id: int):
    """
    Utility to produce an HTML list of games for a given team.
    """
    team_game_objs = get_games_for_team(team_id)
    if not team_game_objs:
        return html.Div("No games submitted yet.")
    items = []
    for g in team_game_objs:
        items.append(html.Li(f"[ID={g.game_id}] {g.title} - {g.image_url or 'No Image'}"))
    return html.Ul(items)

def render_vote_stage(team_id: int):
    """
    Returns layout for A/B voting. We'll randomly pick two games to compare,
    or if there's fewer than 2 games, show a placeholder.
    """
    team_game_objs = get_games_for_team(team_id)
    if len(team_game_objs) < 2:
        return html.Div("Not enough games to vote on.")

    return html.Div([
        html.H4("Pairwise Voting"),
        html.Div(id="vote-pair-container"),
        dbc.Button("Next Pair", id="btn-next-pair", className="mt-3", color="primary"),
        html.Div(id="vote-output", className="mt-2 text-success"),
    ])

@app.callback(
    Output("vote-pair-container", "children"),
    Input("btn-next-pair", "n_clicks"),
    State("team-select", "value")
)
def display_random_pair(n_clicks: int, team_id: int):
    """
    Selects two random distinct games to display side-by-side.
    """
    import random
    team_game_objs = get_games_for_team(team_id)
    if len(team_game_objs) < 2:
        return html.Div("Not enough games to vote on.")

    pair = random.sample(team_game_objs, 2)
    # Simple side by side
    col_style = {"border": "1px solid gray", "borderRadius": "5px", "padding": "10px", "margin": "5px"}

    return dbc.Row([
        dbc.Col([
            html.H5(pair[0].title),
            html.Img(src=pair[0].image_url, style={"width": "100%", "height": "auto"}) if pair[0].image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": pair[0].game_id}, color="success", className="mt-2"),
        ], style=col_style, width=6),

        dbc.Col([
            html.H5(pair[1].title),
            html.Img(src=pair[1].image_url, style={"width": "100%", "height": "auto"}) if pair[1].image_url else None,
            dbc.Button("Choose This", id={"type": "btn-vote-winner", "index": pair[1].game_id}, color="success", className="mt-2"),
        ], style=col_style, width=6),
    ])

@app.callback(
    Output("vote-output", "children"),
    [Input({"type": "btn-vote-winner", "index": dash.ALL}, "n_clicks")],
    [State({"type": "btn-vote-winner", "index": dash.ALL}, "id"),
     State("vote-pair-container", "children"),
     State("team-select", "value")],
    prevent_initial_call=True
)
def handle_vote(n_clicks_list, button_ids, pair_container, team_id: int):
    """
    Records a vote whenever the user clicks on one of the "Choose This" buttons.
    We must parse the *other* game as the loser from the current container.
    """
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    # The clicked button is the winner.
    # We find which button was clicked
    # button_ids is a list of dictionaries with keys {type, index}
    triggered_idx = None
    for i, clicks in enumerate(n_clicks_list):
        if clicks and clicks > 0:
            triggered_idx = i
            break

    if triggered_idx is None:
        raise dash.exceptions.PreventUpdate

    winner_game_id = button_ids[triggered_idx]["index"]

    # Attempt to find the loser by scanning the container
    # (We stored the pair in display_random_pair)
    # We can parse the "index" from each button's ID to see which is the "other" one.
    # For a minimal approach, let's just re-collect them from button_ids
    all_game_ids = [btn["index"] for btn in button_ids]
    if len(all_game_ids) != 2:
        return "Error: Could not identify pair."

    if winner_game_id not in all_game_ids:
        return "Error: Winner not in pair."

    loser_candidates = [gid for gid in all_game_ids if gid != winner_game_id]
    if not loser_candidates:
        return "Error: Could not identify loser."

    loser_game_id = loser_candidates[0]

    record_vote(team_id, winner_game_id, loser_game_id)
    return f"Vote Recorded! Winner={winner_game_id}, Loser={loser_game_id}."

def render_results_stage(team_id: int):
    """
    Returns a layout that displays the computed Bradley-Terry (approx.) scores
    in a sorted manner, plus a bar chart with Plotly.
    """
    scores = compute_bradley_terry_scores(team_id)
    if not scores:
        return html.Div("No votes or not enough games to compute results.")

    # Sort by descending score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Build a small table
    rows = []
    for rank, (g_id, sc) in enumerate(sorted_items, start=1):
        title = games[g_id].title
        rows.append(html.Tr([
            html.Td(rank),
            html.Td(title),
            html.Td(f"{sc:.3f}")
        ]))

    fig = px.bar(
        x=[games[g_id].title for g_id, sc in sorted_items],
        y=[sc for g_id, sc in sorted_items],
        labels={"x": "Game", "y": "Score"},
        title="Estimated Bradley-Terry Scores"
    )

    return html.Div([
        html.H4("Results"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Rank"), html.Th("Game"), html.Th("Score")]))] +
            [html.Tbody(rows)],
            bordered=True
        ),
        dcc.Graph(figure=fig, style={"width": "100%", "height": "500px"}),
        html.Div("Note: Higher score indicates stronger preference in pairwise votes.")
    ])

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Optionally, create a default team to start with:
    default_team_id = create_team("Example Team")
    print(f"Default team created with ID={default_team_id}.")
    app.run_server(debug=True)
