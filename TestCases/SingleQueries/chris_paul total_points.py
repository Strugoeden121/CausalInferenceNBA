import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
from dowhy import CausalModel
from utils.utils import is_treatment_significant

"""
Benchmarks for Usage Rate:
Below 15%: Low usage (usually role players or defensive specialists).
15-20%: Average usage (typical for complementary players).
20-25%: Above average (key contributors or second options).
25-30%: High usage (primary scorers and offensive leaders).
Above 30%: Very high usage (stars or superstars dominating the offense).
Above 35%: Exceptional usage (reserved for players like prime James Harden, Kobe Bryant, or Luka Dončić).
"""

COLUMNS_TO_REMOVE_KOBE = ["winner_team", "assists", "player_id",'player_id_x', 'player_id_y',
                           "season_name", "player_team", 'avg_team_minutes','avg_rival_team_minutes',]

BASE_DIR = "C:/Users/StrugoEden/PycharmProjects/CausalInferenceNBA/Datasets"

DAG_DATA_CP3 = [
    ("rival_team_points", "team_won"),
    ("team_points", "team_won"),
    ("player_name", "minutes"),
    ("minutes", "did_cp3_more_than_13_assists"),
    ("minutes", "points"),
    ("minutes", "rebounds"),
    ("rebounds", "total_team_rebounds"),
    ("did_cp3_more_than_13_assists", "total_team_assists"),
    ("total_rival_team_rebounds", "rival_team_points"),
    ("total_rival_team_assists", "rival_team_points"),
    ("did_cp3_more_than_13_assists", "team_points"),     # Points scored influence the winner
    ("points", "team_points"),
    ("total_team_rebounds", "team_points"),
    ("total_team_assists", "team_points"),]

def augment_chris_paul_dataset():
    duplicated_games_scoring_rivals = pd.read_csv(f"{BASE_DIR}/game_player_stats_final_total.csv")

    chris_paul_data = duplicated_games_scoring_rivals[
        duplicated_games_scoring_rivals['player_name'] == "Chris Paul"]

    # filter for games where 'team_name' column is as 'player_team' column
    chris_paul_data = chris_paul_data[
        chris_paul_data['team_name'] == chris_paul_data["player_team"]]

    # Filter out rows where minutes equals 0
    chris_paul_data = chris_paul_data[chris_paul_data['minutes'] > 0]
    column_name = 'did_cp3_more_than_13_assists'

    chris_paul_data[column_name] = chris_paul_data.apply(
        lambda row: 1 if row['player_name'] == 'Chris Paul' and row['assists'] >= 13 else 0, axis=1)

    print(chris_paul_data[column_name].value_counts())

    #ADD Column for CP3's team in the game

    chris_paul_data["team_won"] = (chris_paul_data["winner_team"] == chris_paul_data["player_team"]).astype(int)

    chris_paul_data = chris_paul_data.drop(columns=COLUMNS_TO_REMOVE_KOBE, errors="ignore")
    print(f"\nDataset shape:{chris_paul_data.shape}")
    print("\nDataset columns:")
    print(chris_paul_data.columns)
    print("\n")
    print(chris_paul_data.head())
    return chris_paul_data


def augment_chris_paul_dag():
    dag_data = list(set(DAG_DATA_CP3))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def chris_paul_model(chris_paul_data, dag):
    model = CausalModel( data=chris_paul_data,treatment='did_cp3_more_than_13_assists', outcome='team_won',graph=dag)

    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand,method_name="backdoor.linear_regression",
        target_units="ate", effect_modifiers=[], test_significance=True)

    print(causal_estimate)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: did_cp3_more_than_13_assists, Outcome: team_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment:  did_cp3_more_than_13_assists, Outcome: team_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')

if __name__ == '__main__':
    print("Treatment - Chris Paul assisting more then 13 assists in game")
    print("Outcome - CP3's Team (changing) Winning the game")
    print("\n")
    chris_paul_dataset = augment_chris_paul_dataset()
    chris_paul_dag = augment_chris_paul_dag()
    chris_paul_model(chris_paul_dataset, chris_paul_dag)