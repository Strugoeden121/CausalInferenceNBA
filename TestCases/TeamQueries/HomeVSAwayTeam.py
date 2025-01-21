import pandas as pd
import os
from utils.utils import BASE_DIR, is_treatment_significant
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel
from utils.constants import TEAM_NAMES_ABBR

COLUMNS_TO_REMOVE = ["winner_team", "home_team", 'away_team', 'did_home_team_win']


def check_team_home_adventage(game_with_teams_dataset, lower_team_name):
    print("Creating Dataset\n")
    team_name = lower_team_name.upper()
    outcome_column_name = f'did_{lower_team_name}_win'
    treatment_column_name = f'is_{lower_team_name}_home_game'

    team_games_dataset = game_with_teams_dataset[(game_with_teams_dataset['home_team'] == team_name)
                                                 | (game_with_teams_dataset['away_team'] == team_name)]
    team_games_dataset[treatment_column_name] = team_games_dataset['winner_team'] == team_name
    team_games_dataset[outcome_column_name] = team_games_dataset['home_team'] == team_name
    team_games_dataset = team_games_dataset.drop(columns=COLUMNS_TO_REMOVE, errors="ignore")

    print("Augmenting DAG\n")

    # Generic game Dag
    # DAG data
    dag_data = [
        (treatment_column_name, "home_points"),
        (treatment_column_name, "away_points"),
        ("away_points", outcome_column_name),
        ("home_points", outcome_column_name),]

    # Remove duplicate edges (if any)
    dag_data = list(set(dag_data))
    dag = nx.DiGraph(dag_data)
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG with adjusted layout
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(dag, k=2)  # Increase k for more sparse node placement
    nx.draw(dag,pos=pos, with_labels=True, node_size=3000,
        node_color="skyblue",font_size=10,font_weight="bold",edge_color="gray")
    plt.title("Causal DAG (Sparse Layout)")
    plt.show()

    print("Calculating Causal Home Adventage\n")

    model = CausalModel(data=team_games_dataset, treatment=treatment_column_name, outcome=outcome_column_name, graph=dag)
    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression",
                                            target_units="ate", effect_modifiers=[], test_significance=True)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: {treatment_column_name}, Outcome: {outcome_column_name}, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: {treatment_column_name}, Outcome: {outcome_column_name}, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')

    return causal_estimate.value


if __name__ == "__main__":
    print("Treatment - Team playing at home")
    print("Outcome - Team Winning the game")
    print("\n")

    game_with_teams_dataset = pd.read_csv(f"{BASE_DIR}/game_with_teams.csv")

    print("First Case - Golden State Warriors")
    check_team_home_adventage(game_with_teams_dataset, "gsw")
    print("Second Case - Utah Jazz")
    check_team_home_adventage(game_with_teams_dataset, "uta")
    print("Third Case - Los Angeles Clippers")
    check_team_home_adventage(game_with_teams_dataset, "lac")
    print("Forth Case - Denver Nuggets")
    check_team_home_adventage(game_with_teams_dataset, "den")
    print("Fifth Case - Sacramento Kings")
    check_team_home_adventage(game_with_teams_dataset, "sac")

    team_to_ate = {}
    sum = 0
    for name in TEAM_NAMES_ABBR:
        team_to_ate[name] = check_team_home_adventage(game_with_teams_dataset, name)
        sum += team_to_ate[name]

    sorted_team_to_ate = dict(sorted(team_to_ate.items(), key=lambda x: x[1], reverse=True))
    print(f"Sorted Teams:{sorted_team_to_ate}")
    print("\n")
    print(f"mean is {sum/len(team_to_ate)}")