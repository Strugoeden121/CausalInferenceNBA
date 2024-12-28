from utils import create_dag_generic
from Preprocessing.generic_single_DAG_creation import DAG_DATA

# function for changing dag node using the new name (treatment and outcome) while removing relevant stat
def augment_dag_list(dag_list_treat, new_treatment_name, stat_name_treatment = '', outcome_team_name = ''):

    for index, (in_edge, out_edge), in enumerate(dag_list_treat):
        if stat_name_treatment and in_edge == f'player_{stat_name_treatment}':
            dag_list_treat[index] = (new_treatment_name, out_edge)
        if stat_name_treatment and out_edge == f'player_{stat_name_treatment}':
            dag_list_treat[index] = (in_edge, new_treatment_name)
        if outcome_team_name and out_edge == 'winner_team':
            dag_list_treat[index] = (in_edge, outcome_team_name)

    create_dag_generic(dag_list_treat)
    return dag_list_treat


if __name__ == '__main__':

    print("Testing augment_dag_list - using steph curry dag'\n")
    new_dag_list = augment_dag_list(DAG_DATA, 'did_steph_scored_more_than_23_points', 'points', 'gsw_won')

    for edge in new_dag_list:
        print(edge)
