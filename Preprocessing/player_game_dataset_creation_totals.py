import pandas as pd
import os


BASE_DIR = "C:/Users/StrugoEden/PycharmProjects/CausalInferenceNBA/Datasets"
# columns to remove - Change if you want more / less columns in final dataset
COLUMNS_TO_REMOVE_PLAYER = [
        "home_id", "offposs", "fg_two_pct", "fg_three_pct", 'fg_two_m', 'fg_two_a', 'fg_three_m',
        'fg_three_a', "ftpoints", "tspct", "usage", "defrebounds", "offrebounds",
        "home_possessions", "away_possessions", "season_type", "away_id", "winner_id",
        "season_id", "team_id"]

COLUMNS_TO_REMOVE_TEAM = [
    "home_team", "away_team", "date_start", "date_end","away_points", "home_points"]

def import_and_print():

    game_df = pd.read_csv(f"{BASE_DIR}/game.csv")
    player_game_stats_df = pd.read_csv(f"{BASE_DIR}/player_game_stats.csv")
    player_salary_df = pd.read_csv(f"{BASE_DIR}/player_salary.csv")
    team_df = pd.read_csv(f"{BASE_DIR}/team.csv")
    season_df = pd.read_csv(f"{BASE_DIR}/season.csv")
    lineup_game_stats_df = pd.read_csv(f"{BASE_DIR}/lineup_game_stats.csv")
    lineup_df = pd.read_csv(f"{BASE_DIR}/lineup.csv")
    lineup_player_df = pd.read_csv(f"{BASE_DIR}/lineup_player.csv")
    play_for_df = pd.read_csv(f"{BASE_DIR}/play_for.csv")
    team_game_stats_df = pd.read_csv(f"{BASE_DIR}/team_game_stats.csv")
    player_df = pd.read_csv(f"{BASE_DIR}/player.csv")
    print(game_df.head())
    print("______________________________________________________________________________________________")
    print(player_game_stats_df.head())
    print("______________________________________________________________________________________________")
    print(lineup_game_stats_df.head())
    print("______________________________________________________________________________________________")
    print(lineup_player_df.head())
    print("______________________________________________________________________________________________")
    print(team_df.head(30))
    print("______________________________________________________________________________________________")
    print(season_df.head(30))
    print("______________________________________________________________________________________________")
    print(player_salary_df.head(30))
    print("______________________________________________________________________________________________")
    print(lineup_df.head(30))
    print("______________________________________________________________________________________________")
    print(team_game_stats_df.head(30))
    print("______________________________________________________________________________________________")
    print(player_df.head(30))
    print("______________________________________________________________________________________________")
    print(play_for_df.head(30))


def create_team_game_data():
    print("Creating team game data.")
    game_df = pd.read_csv(f"{BASE_DIR}/game.csv")
    team_df = pd.read_csv(f"{BASE_DIR}/team.csv")
    game_with_teams = pd.merge(game_df, team_df, left_on="home_id", right_on="team_id", how="left"
    ).rename(columns={"team": "home_team"})

    # Merge away team names
    game_with_teams = pd.merge(
        game_with_teams, team_df, left_on="away_id", right_on="team_id", how="left"
    ).rename(columns={"team": "away_team"})

    # Merge winner team names
    game_with_teams = pd.merge(game_with_teams,team_df, left_on="winner_id", right_on="team_id",
                               how="left").rename(columns={"team": "winner_team"})

    # Drop duplicate columns
    game_with_teams = game_with_teams.drop(columns=["team_id_x", "team_id_y", "team_id"], errors="ignore")
    return game_with_teams

def add_season_data(game_with_teams):
    print("Adding season data.")
    season_df = pd.read_csv(f"{BASE_DIR}/season.csv")
    game_with_season = pd.merge(game_with_teams, season_df, on="season_id", how="left")

    # Preview the resulting DataFrame
    return game_with_season

def add_player_stats(game_with_season):
    print("Adding player stats.")
    player_stats_df = pd.read_csv(f"{BASE_DIR}/player_game_stats.csv")
    play_for_df = pd.read_csv(f"{BASE_DIR}/play_for.csv")
    team_df = pd.read_csv(f"{BASE_DIR}/team.csv")
    player_df = pd.read_csv(f"{BASE_DIR}/player.csv")
    games_scoring = pd.merge(player_stats_df,game_with_season,on=["game_date", "home_id"],
                             how="inner")
    player_with_play_time = pd.merge(play_for_df, player_df, on="player_id", how="inner")
    player_with_team = pd.merge(player_with_play_time, team_df, on="team_id", how="inner")
    player_with_team = player_with_team.rename(columns={"team": "player_team"})
    game_player_stats = pd.merge(games_scoring, player_with_team, on="player_id", how="inner")
    return game_player_stats

def calculate_avg_team_stats(game_player_stats):
    print("Calculating average team stats.")

    # add player_team for game
    filtered_games_scoring = game_player_stats[
        (game_player_stats['game_date'] >= game_player_stats['date_start']) &
        (game_player_stats['game_date'] <= game_player_stats['date_end'])]

    # Averages basic stats
    game_team_averages = filtered_games_scoring.groupby(['game_date', 'player_team']).agg({
        'points': 'sum', 'rebounds': 'sum', 'assists': 'sum', 'minutes': 'sum'}).reset_index()

    game_team_averages.rename(columns={'points': 'total_team_points','rebounds': 'total_team_rebounds',
        'assists': 'total_team_assists', 'minutes': 'total_team_minutes'}, inplace=True)

    # Merge the averages back into the original DataFrame
    player_games_full_dataset = pd.merge(filtered_games_scoring, game_team_averages,
        on=['game_date', 'player_team'], how='left')


    player_games_dataset = player_games_full_dataset.drop(columns=COLUMNS_TO_REMOVE_PLAYER, errors="ignore")
    return player_games_dataset

def add_rival_team_stats(game_player_stats):
    print("Adding rival team stats.")

    # Create a copy of the dataset where team_name is home_team
    home_team_data = game_player_stats.copy()
    home_team_data['team_name'] = home_team_data['home_team']
    home_team_data['rival_team_name'] = home_team_data['away_team']

    # Create a copy of the dataset where team_name is away_team
    away_team_data = game_player_stats.copy()
    away_team_data['team_name'] = away_team_data['away_team']
    away_team_data['rival_team_name'] = away_team_data['home_team']
    # Concatenate the two datasets to get twice the rows
    duplicated_game_player_stats = pd.concat([home_team_data, away_team_data], ignore_index=True)
    # Add team_points and rival_team_points
    duplicated_game_player_stats['team_points'] = duplicated_game_player_stats.apply(
        lambda row: row['home_points'] if row['team_name'] == row['home_team'] else row['away_points'], axis=1)

    duplicated_game_player_stats['rival_team_points'] = duplicated_game_player_stats.apply(
        lambda row: row['away_points'] if row['team_name'] == row['home_team'] else row['home_points'], axis=1)
    duplicated_game_player_stats = duplicated_game_player_stats.drop(columns=COLUMNS_TO_REMOVE_TEAM, errors="ignore")

    duplicated_game_player_stats['rival_player_team'] = duplicated_game_player_stats.apply(
        lambda row: row['rival_team_name'] if row['player_team'] == row['team_name'] else row['team_name'], axis=1)

    # Add the rival_player_team column based on the condition

    rival_game_team_averages = duplicated_game_player_stats.groupby(['game_date', 'rival_player_team']).agg({
        'points': 'sum','rebounds': 'sum','assists': 'sum','minutes': 'sum'}).reset_index()

    # Rename columns for clarity
    rival_game_team_averages.rename(columns={ 'points': 'total_rival_team_points','rebounds': 'total_rival_team_rebounds',
        'assists': 'total_rival_team_assists','minutes': 'total_rival_team_minutes'}, inplace=True)

    # Merge the averages back into the original DataFrame
    duplicated_games_scoring_rivals = pd.merge(duplicated_game_player_stats,
        rival_game_team_averages, on=['game_date', 'rival_player_team'],how='left')


    # View the updated DataFrame
    print(duplicated_games_scoring_rivals.head())
    print(duplicated_games_scoring_rivals.shape)
    print(duplicated_games_scoring_rivals.columns)
    return duplicated_games_scoring_rivals


if __name__ == "__main__":

    game_with_teams = create_team_game_data()
    game_with_season = add_season_data(game_with_teams)
    game_player_stats_original = add_player_stats(game_with_season)
    game_player_stats2 = calculate_avg_team_stats(game_player_stats_original)
    game_player_stats2.to_csv(f"{BASE_DIR}/game_player_stats_first_total.csv", index=False)
    game_player_stats_final = add_rival_team_stats(game_player_stats2)
    game_player_stats_final.to_csv(f"{BASE_DIR}/game_player_stats_final_total.csv", index=False)
