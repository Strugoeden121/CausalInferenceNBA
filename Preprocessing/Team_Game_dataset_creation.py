import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt


BASE_DIR = "C:/Users/StrugoEden/PycharmProjects/CausalInferenceNBA/Datasets"
COLUMNS_TO_REMOVE_PLAYER = ["team_id_x", "team_id_y", "team_id", "season_id", "winner_id", "away_id", "home_id",
     "away_possessions", "home_possessions"]

def create_team_game_dataset():
    team_df = pd.read_csv(f"{BASE_DIR}/team.csv")
    game_df = pd.read_csv(f"{BASE_DIR}/game.csv")

    game_with_teams = pd.merge(game_df, team_df, left_on="home_id",
        right_on="team_id", how="left").rename(columns={"team": "home_team"})
    game_with_teams = pd.merge(game_with_teams, team_df, left_on="away_id",
        right_on="team_id", how="left").rename(columns={"team": "away_team"})

    game_with_teams = pd.merge(game_with_teams,team_df,left_on="winner_id",
        right_on="team_id",how="left" ).rename(columns={"team": "winner_team"})

    game_with_teams = game_with_teams.drop(columns=COLUMNS_TO_REMOVE_PLAYER, errors="ignore")
    game_with_teams['did_home_team_win'] = game_with_teams['home_team'] == game_with_teams['winner_team']
    print(game_with_teams.head())
    game_with_teams.to_csv(f"{BASE_DIR}/game_with_teams.csv", index=False)

def create_team_game_dag():
    # Remove duplicate edges (if any)

    dag_data = [
        ("home_team", "home_points"),
        ("away_team", "away_points"),
        ("away_points", "winner_team"),
        ("home_points", "winner_team"), ]
    dag_data = list(set(dag_data))

    # Create a DAG
    dag = nx.DiGraph(dag_data)

    # Check if the graph is a DAG
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray",
            pos=nx.spring_layout(dag))  # Spring layout for better visualization

    plt.title("Causal DAG")
    plt.show()


if __name__ == "__main__":
    create_team_game_dataset()
    create_team_game_dag()