import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from utils.utils import BASE_DIR

def create_due_dag():
    dag_data = [
        ("player_name", "player_1_minutes"),
        ("player_1_minutes", "player_1_points"),
        ("player_1_minutes", "player_1_assists"),
        ("player_1_minutes", "player_1_rebounds"),
        ("player_name_2", "player_2_minutes"),
        ("player_2_minutes", "player_2_points"),
        ("player_2_minutes", "player_2_assists"),
        ("player_2_minutes", "player_2_rebounds"),
        ("player_1_points", "avg_team_points"),  # More minutes played may lead to more points
        ("player_1_rebounds", "avg_team_rebounds"),
        ("player_1_assists", "avg_team_assists"),
        ("player_2_assists", "avg_team_assists"),
        ("player_2_points", "avg_team_points"),  # More minutes played may lead to more points
        ("player_2_rebounds", "avg_team_rebounds"),
        ("avg_rival_team_points", "rival_team_points"),  # Points scored influence the winner
        ("avg_rival_team_rebounds", "rival_team_points"),
        ("avg_rival_team_assists", "rival_team_points"),
        ("player_1_points", "team_points"),
        ("player_2_points", "team_points"),
        ("avg_team_points", "team_points"),  # Points scored influence the winner
        ("avg_team_rebounds", "team_points"),
        ("avg_team_assists", "team_points"),
        ("rival_team_points", "winner_team"),
        ("team_points", "winner_team"),    ]

    dag_data = list(set(dag_data))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray", pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def created_steph_kley_dataset():
    print("Creating steph kley dataset...")
    single_game_player = pd.read_csv(f"{BASE_DIR}/game_player_stats_final.csv")

    #Filer for games with steph and kley
    steph_kley_data = single_game_player[
        single_game_player['player_name'].isin(["Stephen Curry", "Klay Thompson"])]
    steph_kley_data_cleaned = steph_kley_data.drop_duplicates(subset=["game_date", "player_name"])

    # Filter games where both Stephen Curry and Klay Thompson played
    games_both_played_steph_kley = steph_kley_data_cleaned.groupby('game_date').filter(
        lambda x: set(["Stephen Curry", "Klay Thompson"]).issubset(set(x['player_name'])))

    #Change player stat names
    for stat in ['points', 'rebounds', 'assists', 'minutes']:
        games_both_played_steph_kley.rename(columns={stat: f"player_1_{stat}" if "Stephen Curry" in
        games_both_played_steph_kley['player_name'].values else f"player_2_{stat}"}, inplace=True)

    # Filter the dataset for Klay Thompson and Stephen Curry
    klay_data = games_both_played_steph_kley[games_both_played_steph_kley['player_name'] == 'Klay Thompson']
    steph_data = games_both_played_steph_kley[games_both_played_steph_kley['player_name'] == 'Stephen Curry']

    klay_data = klay_data.rename( columns={ "player_1_minutes": "player_2_minutes", "player_1_points": "player_2_points",
            "player_1_assists": "player_2_assists", "player_1_rebounds": "player_2_rebounds","player_name": "player_name_2"})

    #take only kley's relevant stats
    klay_subset = klay_data[
        ["game_date", "player_2_minutes", "player_2_points", "player_2_assists", "player_2_rebounds", "player_name_2"]]

    merged_data = pd.merge(steph_data, klay_subset, on="game_date", how="left")

    merged_data["gsw_won"] = (merged_data["winner_team"] == "GSW").astype(int)
    print(f"\nDataset shape:{merged_data.shape}")
    print("\nDataset columns:")
    print(merged_data.columns)
    print(merged_data.head())
    merged_data.to_csv(f"{BASE_DIR}/steph_kley_dataset.csv", index=False)
    create_due_dag()


def created_durent_westbrook_dataset():
    print("Creating Durent Westbrook dataset...")
    single_game_player = pd.read_csv(f"{BASE_DIR}/game_player_stats_final.csv")

    #Filer for games with durent and westbrook
    durent_westbrook_data = single_game_player[
        single_game_player['player_name'].isin(["Kevin Durant", "Russell Westbrook"])]
    durent_westbrook_data_cleaned = durent_westbrook_data.drop_duplicates(subset=["game_date", "player_name"])

    # Filter games where both Kevin Durant and Russell Westbrook played
    games_both_played_durent_westbrook = durent_westbrook_data_cleaned.groupby('game_date').filter(
        lambda x: set(["Kevin Durant", "Russell Westbrook"]).issubset(set(x['player_name'])))

    #Change player stat names
    for stat in ['points', 'rebounds', 'assists', 'minutes']:
        games_both_played_durent_westbrook.rename(columns={stat: f"player_1_{stat}" if "Russell Westbrook" in
        games_both_played_durent_westbrook['player_name'].values else f"player_2_{stat}"}, inplace=True)

    # Filter the dataset for Russell Westbrook and Kevin Durant
    durent_data = games_both_played_durent_westbrook[games_both_played_durent_westbrook['player_name'] == 'Kevin Durant']
    westbrook_data = games_both_played_durent_westbrook[games_both_played_durent_westbrook['player_name'] == 'Russell Westbrook']

    durent_data = durent_data.rename( columns={ "player_1_minutes": "player_2_minutes", "player_1_points": "player_2_points",
            "player_1_assists": "player_2_assists", "player_1_rebounds": "player_2_rebounds","player_name": "player_name_2"})

    #take only durent's relevant stats
    durent_subset = durent_data[
        ["game_date", "player_2_minutes", "player_2_points", "player_2_assists", "player_2_rebounds", "player_name_2"]]

    merged_data = pd.merge(westbrook_data, durent_subset, on="game_date", how="left")

    merged_data["okc_won"] = (merged_data["winner_team"] == "OKC").astype(int)
    print(f"\nDataset shape:{merged_data.shape}")
    print("\nDataset columns:")
    print(merged_data.columns)
    print(merged_data.head())
    merged_data.to_csv(f"{BASE_DIR}/durent_westbrook_dataset.csv", index=False)
    create_due_dag()



if __name__ == "__main__":
    #created_steph_kley_dataset()
    created_durent_westbrook_dataset()