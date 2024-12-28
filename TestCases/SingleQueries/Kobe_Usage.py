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

COLUMNS_TO_REMOVE_KOBE = ["winner_team", "points", "player_id",'player_id_x', 'player_id_y',
                           "season_name", "player_team", 'avg_team_minutes','avg_rival_team_minutes',]

BASE_DIR = "C:/Users/StrugoEden/PycharmProjects/CausalInferenceNBA/Datasets"

DAG_DATA_KOBE = [
    ("rival_team_points", "lal_won"),
    ("team_points", "lal_won"),
    ("player_name", "minutes"),
    ("minutes", "assists"),
    ("minutes", "did_kobe_more_than_30_points"),
    ("minutes", "rebounds"),
    ("rebounds", "avg_team_rebounds"),
    ("assists", "avg_team_assists"),
    ("avg_rival_team_points", "rival_team_points"),     # Points scored influence the winner
    ("avg_rival_team_rebounds", "rival_team_points"),
    ("avg_rival_team_assists", "rival_team_points"),
    ("did_kobe_more_than_30_points", "team_points"),     # Points scored influence the winner
    ("did_kobe_more_than_30_points", "avg_team_points"),
    ("avg_team_points", "team_points"),
    ("avg_team_rebounds", "team_points"),
    ("avg_team_assists", "team_points"),]


def augment_kobe_bryant_dataset():
    duplicated_games_scoring_rivals = pd.read_csv(f"{BASE_DIR}/game_player_stats_final.csv")

    kobe_bryant_data = duplicated_games_scoring_rivals[
        duplicated_games_scoring_rivals['player_name'] == "Kobe Bryant"]

    # Filter out rows where minutes equals 0
    kobe_bryant_data = kobe_bryant_data[kobe_bryant_data['minutes'] > 0]
    column_name = 'did_kobe_more_than_30_points'

    kobe_bryant_data[column_name] = kobe_bryant_data.apply(
        lambda row: 1 if row['player_name'] == 'Kobe Bryant' and row['points'] >= 30 else 0, axis=1)

    print(kobe_bryant_data[column_name].value_counts())

    # Create a new column 'lal_won' indicating if LAL won the game
    kobe_bryant_data["lal_won"] = (kobe_bryant_data["winner_team"] == "LAL").astype(int)
    kobe_bryant_data = kobe_bryant_data.drop(columns=COLUMNS_TO_REMOVE_KOBE, errors="ignore")
    print(f"\nDataset shape:{kobe_bryant_data.shape}")
    print("\nDataset columns:")
    print(kobe_bryant_data.columns)
    print("\n")
    print(kobe_bryant_data.head())
    return kobe_bryant_data


def augment_kobe_dag():
    dag_data = list(set(DAG_DATA_KOBE))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def kobe_causal_model(kobe_bryant_data, dag):
    model = CausalModel( data=kobe_bryant_data,treatment='did_kobe_more_than_30_points', outcome='lal_won',graph=dag)

    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand,method_name="backdoor.linear_regression",
        target_units="ate", effect_modifiers=[], test_significance=True)

    print(causal_estimate)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: did_kobe_more_than_30_points, Outcome: lal_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment:  did_kobe_more_than_30_points, Outcome: lal_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')

if __name__ == '__main__':
    print("Treatment - Kobe Bryant scoring more then 30 points in game")
    print("Outcome - Los Angeles Lakers Winning the game")
    print("\n")
    kobe_bryant_dataset = augment_kobe_bryant_dataset()
    kobe_dag = augment_kobe_dag()
    kobe_causal_model(kobe_bryant_dataset, kobe_dag)