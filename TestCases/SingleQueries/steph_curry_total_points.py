import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
from dowhy import CausalModel
from utils.utils import is_treatment_significant, BASE_DIR


COLUMNS_TO_REMOVE_STEPH = ["winner_team", "assists", "rebounds", "points","minutes", "player_id",
    "season_name", "player_team"]


DAG_DATA_STEPH = [
    ("rival_team_points", "gsw_won"),
    ("team_points", "gsw_won"),
    ("player_name", "player_minutes"),
    ("player_minutes", "player_assists"),
    ("player_minutes", "did_steph_scored_more_than_23_points"),
    ("player_minutes", "player_rebounds"),
    ("player_rebounds", "total_team_rebounds"),
    ("player_assists", "total_team_assists"),
    ("total_rival_team_rebounds", "rival_team_points"),
    ("total_rival_team_assists", "rival_team_points"),
    ("did_steph_scored_more_than_23_points", "team_points"),     # Points scored influence the winner
    ("total_team_rebounds", "team_points"),
    ("total_team_assists", "team_points"),]

def augment_steph_curry_dataset():
    duplicated_games_scoring_rivals = pd.read_csv(f"{BASE_DIR}/game_player_stats_final_total.csv")

    steph_curry_data = duplicated_games_scoring_rivals[
        duplicated_games_scoring_rivals['player_name'] == "Stephen Curry"]

    steph_curry_data.to_csv(f"{BASE_DIR}/steph_curry_data.csv", index=False)

    #filter for games where 'team_name' column is as 'player_team' column
    steph_curry_data = steph_curry_data[
        steph_curry_data['team_name'] == steph_curry_data["player_team"]]

    # Filter out rows where minutes equals 0
    steph_curry_data = steph_curry_data[steph_curry_data['minutes'] > 0]

    # Add a column to indicate if Steph Curry scored more than 23 points
    steph_curry_data['did_steph_scored_more_than_23_points'] = steph_curry_data.apply(
        lambda row: 1 if row['player_name'] == 'Stephen Curry' and row['points'] >= 23 else 0, axis=1)

    print(steph_curry_data['did_steph_scored_more_than_23_points'].value_counts())

    # Create a new column 'gsw_won' indicating if GSW won the game
    steph_curry_data["gsw_won"] = (steph_curry_data["winner_team"] == "GSW").astype(int)
    steph_curry_data = steph_curry_data.drop(columns=COLUMNS_TO_REMOVE_STEPH, errors="ignore")
    print(f"\nDataset shape:{steph_curry_data.shape}")
    print("\nDataset columns:")
    print(steph_curry_data.columns)
    print("\n")
    print(steph_curry_data.head())
    return steph_curry_data


def augment_steph_dag():
    dag_data = list(set(DAG_DATA_STEPH))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()
    return dag

def steph_causal_model(dag, steph_curry_data):
    # Causal model
    model = CausalModel(
        data=steph_curry_data,
        treatment='did_steph_scored_more_than_23_points',
        outcome='gsw_won',
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
        print(f'Treatment: steph_scored_23_plus, Outcome: gsw_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: steph_scored_23_plus, Outcome: gsw_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')


if __name__ == '__main__':
    print("Treatment - Steph Curry scoring more then 23 points in game")
    print("Outcome - Golden State Warriors Winning the game")
    print("\n")
    steph_curry_dataset = augment_steph_curry_dataset()
    steph_dag = augment_steph_dag()
    steph_causal_model(steph_dag, steph_curry_dataset)




