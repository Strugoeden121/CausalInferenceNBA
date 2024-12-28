import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
from dowhy import CausalModel
from utils.utils import is_treatment_significant, BASE_DIR


COLUMNS_TO_REMOVE_russell = ["winner_team", "assists", "rebounds", "points","minutes", "player_id",
    "season_name", "player_team"]


DAG_DATA_russell = [
    ("rival_team_points", "okc_won"),
    ("team_points", "okc_won"),
    ("player_name", "player_minutes"),
    ("player_minutes", "player_assists"),
    ("player_minutes", "did_russell_scored_more_than_23_points"),
    ("player_minutes", "player_rebounds"),
    ("player_rebounds", "avg_team_rebounds"),
    ("player_assists", "avg_team_assists"),
    ("avg_rival_team_points", "rival_team_points"),     # Points scored influence the winner
    ("avg_rival_team_rebounds", "rival_team_points"),
    ("avg_rival_team_assists", "rival_team_points"),
    ("did_russell_scored_more_than_23_points", "team_points"),     # Points scored influence the winner
    ("did_russell_scored_more_than_23_points", "avg_team_points"),
    ("avg_team_points", "team_points"),
    ("avg_team_rebounds", "team_points"),
    ("avg_team_assists", "team_points"),]

def augment_russell_westbrook_dataset():
    duplicated_games_scoring_rivals = pd.read_csv(f"{BASE_DIR}/game_player_stats_final.csv")

    russell_westbrook_data = duplicated_games_scoring_rivals[
        duplicated_games_scoring_rivals['player_name'] == "Russell Westbrook"]

    # Filter out rows where minutes equals 0
    russell_westbrook_data = russell_westbrook_data[russell_westbrook_data['minutes'] > 0]

    # Add a column to indicate if russell westbrook scored more than 23 points
    russell_westbrook_data['did_russell_scored_more_than_23_points'] = russell_westbrook_data.apply(
        lambda row: 1 if row['player_name'] == 'Russell Westbrook' and row['points'] >= 23 else 0, axis=1)

    print(russell_westbrook_data['did_russell_scored_more_than_23_points'].value_counts())

    # Create a new column 'okc_won' indicating if okc won the game
    russell_westbrook_data["okc_won"] = (russell_westbrook_data["winner_team"] == "OKC").astype(int)
    russell_westbrook_data = russell_westbrook_data.drop(columns=COLUMNS_TO_REMOVE_russell, errors="ignore")
    print(f"\nDataset shape:{russell_westbrook_data.shape}")
    print("\nDataset columns:")
    print(russell_westbrook_data.columns)
    print("\n")
    print(russell_westbrook_data.head())
    return russell_westbrook_data


def augment_russell_dag():
    dag_data = list(set(DAG_DATA_russell))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def russell_causal_model(dag, russell_westbrook_data):
    # Causal model
    model = CausalModel(
        data=russell_westbrook_data,
        treatment='did_russell_scored_more_than_23_points',
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
        print(f'Treatment: russell_scored_23_plus, Outcome: okc_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: russell_scored_23_plus, Outcome: okc_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')


if __name__ == '__main__':
    print("Treatment - Russel Westbrook scoring more then 23 points in game")
    print("Outcome - OKC Winning the game")
    print("\n")
    russell_westbrook_dataset = augment_russell_westbrook_dataset()
    russell_dag = augment_russell_dag()
    russell_causal_model(russell_dag, russell_westbrook_dataset)




