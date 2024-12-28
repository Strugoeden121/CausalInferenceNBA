import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
from dowhy import CausalModel
from utils.utils import is_treatment_significant, BASE_DIR


COLUMNS_TO_REMOVE_durent = ["winner_team", "assists", "rebounds", "points","minutes", "player_id",
    "season_name", "player_team"]


DAG_DATA_durent = [
    ("rival_team_points", "okc_won"),
    ("team_points", "okc_won"),
    ("player_name", "player_minutes"),
    ("player_minutes", "player_assists"),
    ("player_minutes", "did_durent_scored_more_than_25_points"),
    ("player_minutes", "player_rebounds"),
    ("player_rebounds", "avg_team_rebounds"),
    ("player_assists", "avg_team_assists"),
    ("avg_rival_team_points", "rival_team_points"),     # Points scored influence the winner
    ("avg_rival_team_rebounds", "rival_team_points"),
    ("avg_rival_team_assists", "rival_team_points"),
    ("did_durent_scored_more_than_25_points", "team_points"),     # Points scored influence the winner
    ("did_durent_scored_more_than_25_points", "avg_team_points"),
    ("avg_team_points", "team_points"),
    ("avg_team_rebounds", "team_points"),
    ("avg_team_assists", "team_points"),]

def augment_kd_dataset():
    duplicated_games_scoring_rivals = pd.read_csv(f"{BASE_DIR}/game_player_stats_final.csv")

    kd_data = duplicated_games_scoring_rivals[
        duplicated_games_scoring_rivals['player_name'] == "Kevin Durant"]

    # Filter out rows where minutes equals 0
    kd_data = kd_data[kd_data['minutes'] > 0]

    # Add a column to indicate if Kevin Durant scored more than 25 points
    kd_data['did_durent_scored_more_than_25_points'] = kd_data.apply(
        lambda row: 1 if row['player_name'] == 'Kevin Durant' and row['points'] >= 25 else 0, axis=1)

    print(kd_data['did_durent_scored_more_than_25_points'].value_counts())

    # Create a new column 'okc_won' indicating if okc won the game
    kd_data["okc_won"] = (kd_data["winner_team"] == "OKC").astype(int)
    kd_data = kd_data.drop(columns=COLUMNS_TO_REMOVE_durent, errors="ignore")
    print(f"\nDataset shape:{kd_data.shape}")
    print("\nDataset columns:")
    print(kd_data.columns)
    print("\n")
    print(kd_data.head())
    return kd_data


def augment_durent_dag():
    dag_data = list(set(DAG_DATA_durent))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def durent_causal_model(dag, kd_data):
    # Causal model
    model = CausalModel(
        data=kd_data,
        treatment='did_durent_scored_more_than_25_points',
        outcome='okc_won',
        graph=dag
    )

    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
        effect_modifiers=[],
        test_significance=True
    )

    print(causal_estimate)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: durent_scored_25_plus, Outcome: okc_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: durent_scored_25_plus, Outcome: okc_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')


if __name__ == '__main__':
    print("Treatment - Kevin Durant scoring more then 25 points in game")
    print("Outcome - OKC Winning the game")
    print("\n")
    kd_dataset = augment_kd_dataset()
    durent_dag = augment_durent_dag()
    durent_causal_model(durent_dag, kd_dataset)




