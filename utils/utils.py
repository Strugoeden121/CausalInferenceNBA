import matplotlib.pyplot as plt
import networkx as nx
from dowhy import CausalModel

BASE_DIR = "C:/Users/StrugoEden/PycharmProjects/CausalInferenceNBA/Datasets"

def nodes_to_remove_from_dag(dag_data = [], nodes_to_remove = []):
    new_dag = []
    for source, dest in dag_data:
        if source in nodes_to_remove or dest in nodes_to_remove:
            continue
        new_dag.append((source, dest))

    return new_dag

def is_treatment_significant(causal_estimate):
    p_value = causal_estimate.test_stat_significance()['p_value']
    return p_value is not None and p_value < 0.05

def create_dag_generic(dag_data):
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

def calculate_causal_effect_generic(dataset, dag, treatment, outcome):
    # Causal model
    model = CausalModel(data=dataset, treatment=treatment, outcome=outcome, graph=dag)
    # Identify and estimate the effect
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression",
                                            target_units="ate", effect_modifiers=[], test_significance=True)

    if is_treatment_significant(causal_estimate):
        print(f'Treatment: {treatment}, Outcome: {outcome}, ATE: {causal_estimate.value}')
    else:
        print(
            f'Treatment: {treatment}, Outcome: {outcome}, ATE: Not significant, p-value: {causal_estimate.test_stat_significance()["p_value"]}')
