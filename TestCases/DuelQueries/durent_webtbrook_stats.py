import pandas as pd
import os
from utils.utils import BASE_DIR, is_treatment_significant
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel

COLUMNS_TO_REMOVE = ["player_id", "season_name", 'winner_team', 'team_name']

def calculate_causal_durent_westbrook_ass(durent_westbrook_dataset, westbrook_assists, durent_points):
    print("creating Dataset")
    column_name = f'westbrook_{westbrook_assists}ass+_durent_{durent_points}+'
    durent_westbrook_dataset[column_name] = (
                (durent_westbrook_dataset['player_1_assists'] >= westbrook_assists) & (durent_westbrook_dataset['player_2_points'] >= durent_points)).astype(int)
    column_counts = durent_westbrook_dataset[column_name].value_counts()
    print(column_counts)

    # Drop the specified columns
    durent_westbrook_dataset = durent_westbrook_dataset.drop(columns=COLUMNS_TO_REMOVE, errors="ignore")

    print("Augmenting DAG")
    dag_data = [
        ("player_name", "player_1_minutes"),
        ("player_1_minutes", 'player_1_points'),
        ("player_1_minutes", column_name),
        ("player_1_minutes", "player_1_rebounds"),
        ("player_name_2", "player_2_minutes"),
        ("player_2_minutes", column_name),
        ("player_2_minutes", 'player_2_assists'),
        ("player_2_minutes", "player_2_rebounds"),
        ('player_1_points', "avg_team_points"),  # More minutes played may lead to more points
        ("player_1_rebounds", "avg_team_rebounds"),
        (column_name, "avg_team_assists"),
        ("player_2_assists", "avg_team_assists"),
        (column_name, "avg_team_points"),  # More minutes played may lead to more points
        ("player_2_rebounds", "avg_team_rebounds"),
        ("avg_rival_team_points", "rival_team_points"),  # Points scored influence the winner
        ("avg_rival_team_rebounds", "rival_team_points"),
        ("avg_rival_team_assists", "rival_team_points"),
        ('player_1_points', "team_points"),
        (column_name, "team_points"),
        ("avg_team_points", "team_points"),  # Points scored influence the winner
        ("avg_team_rebounds", "team_points"),
        ("avg_team_assists", "team_points"),
        ("rival_team_points", "okc_won"),
        ("team_points", "okc_won"), ]

    dag_data = list(set(dag_data))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray", pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()

    print("Calculating causal westbrook durent")
    # Causal model
    model = CausalModel(data=durent_westbrook_dataset, treatment=column_name,outcome='okc_won',graph=dag)
    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression",
                                            target_units="ate", effect_modifiers=[], test_significance=True)

    print(causal_estimate)
    if is_treatment_significant(causal_estimate):
        print(f'Treatment: {column_name}, Outcome: okc_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: {column_name}, Outcome: okc_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')


def calculate_causal_durent_westbrook(durent_westbrook_dataset, westbrook_points, durent_points):
    print("Creating Dataset")
    column_name = f'westbrook_{westbrook_points}+_durent_{durent_points}+'
    durent_westbrook_dataset[column_name] = (
                (durent_westbrook_dataset['player_2_points'] >= durent_points) & (durent_westbrook_dataset['player_1_points'] >= westbrook_points)).astype(int)
    column_counts = durent_westbrook_dataset[column_name].value_counts()
    print(column_counts)

    # Drop the specified columns
    durent_westbrook_dataset = durent_westbrook_dataset.drop(columns=COLUMNS_TO_REMOVE, errors="ignore")

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
        (column_name, "avg_team_points"),  # More minutes played may lead to more points
        ("player_1_rebounds", "avg_team_rebounds"),
        ("player_1_assists", "avg_team_assists"),
        ("player_2_assists", "avg_team_assists"),
        (column_name, "avg_team_points"),  # More minutes played may lead to more points
        ("player_2_rebounds", "avg_team_rebounds"),
        ("avg_rival_team_points", "rival_team_points"),  # Points scored influence the winner
        ("avg_rival_team_rebounds", "rival_team_points"),
        ("avg_rival_team_assists", "rival_team_points"),
        (column_name, "team_points"),
        (column_name, "team_points"),
        ("avg_team_points", "team_points"),  # Points scored influence the winner
        ("avg_team_rebounds", "team_points"),
        ("avg_team_assists", "team_points"),
        ("rival_team_points", "okc_won"),
        ("team_points", "okc_won"), ]

    dag_data = list(set(dag_data))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color="gray", pos=nx.spring_layout(dag))

    plt.title("Causal DAG")
    plt.show()

    print("Calculating causal westbrook durent")
    # Causal model
    model = CausalModel(data=durent_westbrook_dataset, treatment=column_name,outcome='okc_won',graph=dag)
    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect( identified_estimand,method_name="backdoor.linear_regression",
                                             target_units="ate", effect_modifiers=[],test_significance=True)


    if is_treatment_significant(causal_estimate):
        print(f'Treatment: {column_name}, Outcome: okc_won, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: {column_name}, Outcome: okc_won, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')





if __name__ == "__main__":
    print("Treatment - westbrook scoring more then X points and durent Scoring more then Y points in game")
    print("Outcome - OKC Winning the game")
    print("\n")

    durent_westbrook_dataset = pd.read_csv(f"{BASE_DIR}/durent_westbrook_dataset.csv")

    print("First Case - westbrook 20+ points durent 15+ points")
    calculate_causal_durent_westbrook(durent_westbrook_dataset, westbrook_points=20, durent_points=15)
    print("Second Case - westbrook 15+ points durent 20+ points")
    calculate_causal_durent_westbrook(durent_westbrook_dataset, westbrook_points=15, durent_points=20)
    print("Third Case - westbrook 20+ points durent 20+ points")
    calculate_causal_durent_westbrook(durent_westbrook_dataset, westbrook_points=20, durent_points=20)
    print("Forth Case - westbrook 15+ points durent 15+ points")
    calculate_causal_durent_westbrook(durent_westbrook_dataset, westbrook_points=15, durent_points=15)

    print("--------------------------------------------------------")
    print("Treatment - westbrook having more then X assists and durent Scoring more then Y points in game")
    print("Outcome - OKC Winning the game")
    print("\n")
    print("First Case - westbrook 10+ assists durent 23+ points")
    calculate_causal_durent_westbrook_ass(durent_westbrook_dataset, westbrook_assists=10, durent_points=23)

