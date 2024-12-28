import networkx as nx
import matplotlib.pyplot as plt

DAG_DATA = [("player_name", "player_minutes"),
            ("player_minutes", "player_assists"),
            ("player_minutes", "player_points"),
            ("player_minutes", "player_rebounds"),
            ("player_rebounds", "total_team_rebounds"),
            ("player_assists", "total_team_assists"),
            ("player_points", "team_points"),
            ("total_rival_team_rebounds", "rival_team_points"),
            ("total_rival_team_assists", "rival_team_points"),
            ("total_team_rebounds", "team_points"),
            ("total_team_assists", "team_points"),
            ("rival_team_points", "winner_team"),
            ("team_points", "winner_team"),]

if __name__ == "__main__":


    # Remove duplicate edges (if any)
    dag_data = list(set(DAG_DATA))

    # Create a DAG
    dag = nx.DiGraph(dag_data)

    # Check if the graph is a DAG
    print(f"Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(dag)}")

    # Draw the DAG
    plt.figure(figsize=(10, 8))
    nx.draw(dag, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10,font_weight="bold",edge_color="gray",
        pos=nx.spring_layout(dag))  # Spring layout for better visualization

    plt.title("Causal DAG")
    plt.show()