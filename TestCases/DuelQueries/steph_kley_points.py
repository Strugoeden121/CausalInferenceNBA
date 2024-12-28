import pandas as pd
import os
from utils.utils import BASE_DIR, is_treatment_significant
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel

COLUMNS_TO_REMOVE = ["player_id", "season_name", 'winner_team', 'team_name']



def calculate_causal_steph_kley(steph_kley_dataset, steph_points, kley_points):
    print("creating Dataset")
    column_name = f'steph_{steph_points}+_kley_{kley_points}+'
    steph_kley_dataset[column_name] = (
                (steph_kley_dataset['player_2_points'] >= steph_points) & (steph_kley_dataset['player_1_points'] >= kley_points)).astype(int)
    column_counts = steph_kley_dataset[column_name].value_counts()
    print(column_counts)

    # Drop the specified columns
    steph_kley_dataset = steph_kley_dataset.drop(columns=COLUMNS_TO_REMOVE, errors="ignore")

    print("Augmenting DAG")
    dag_data = [
        ("player_name", "player_1_minutes"),
        ("player_1_minutes", column_name),
        ("player_1_minutes", "player_1_assists"),
        ("player_1_minutes", "player_1_rebounds"),
        ("player_name_2", "player_2_minutes"),
        ("player_2_minutes", column_name),
        ("player_2_minutes", "player_2_assists"),
        ("player_2_minutes", "player_2_rebounds"),
        ("rival_team_points", "gsw_won"),
        ("team_points", "gsw_won"),
        (column_name, "avg_team_points"),  # More minutes played may lead to more points
        ("player_1_rebounds", "avg_team_rebounds"),
        ("player_2_assists", "avg_team_assists"),
        ("player_2_rebounds", "avg_team_rebounds"),
        ("player_1_assists", "avg_team_assists"),
        ("avg_rival_team_points", "rival_team_points"),  # Points scored influence the winner
        ("avg_rival_team_rebounds", "rival_team_points"),
        ("avg_rival_team_assists", "rival_team_points"),
        (column_name, "team_points"),
        ("avg_team_points", "team_points"),  # Points scored influence the winner
        ("avg_team_rebounds", "team_points"),
        ("avg_team_assists", "team_points"),
    ]

    dag_data = list(set(dag_data))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray", pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()

    print("Calculating causal steph kley")
    # Causal model
    model = CausalModel(data=steph_kley_dataset, treatment=column_name,outcome='gsw_won',graph=dag)

    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect( identified_estimand,method_name="backdoor.linear_regression",
                                             target_units="ate", effect_modifiers=[],test_significance=True)

    def is_treatment_significant(causal_estimate):
        p_value = causal_estimate.test_stat_significance()['p_value']
        return p_value is not None and p_value < 0.05

    print(causal_estimate)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: {column_name}, Outcome: gsw_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: {column_name}, Outcome: gsw_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')



if __name__ == "__main__":
    print("Treatment - Steph Curry scoring more then X points and Kley Scoring more then Y points in game")
    print("Outcome - Golden State Warriors Winning the game")
    print("\n")

    steph_kley_dataset = pd.read_csv(f"{BASE_DIR}/steph_kley_dataset.csv")

    print("First Case - Steph 20+ points Kley 15+ points")
    calculate_causal_steph_kley(steph_kley_dataset, steph_points=23, kley_points=20)
    print("Second Case - Steph 15+ points Kley 20+ points")
    calculate_causal_steph_kley(steph_kley_dataset, steph_points=15, kley_points=20)
    print("Third Case - Steph 20+ points Kley 20+ points")
    calculate_causal_steph_kley(steph_kley_dataset, steph_points=20, kley_points=20)
    print("Forth Case - Steph 15+ points Kley 15+ points")
    calculate_causal_steph_kley(steph_kley_dataset, steph_points=15, kley_points=15)