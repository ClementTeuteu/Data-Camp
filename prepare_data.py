### Link to the dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores

#### Pre-processing the dataset

#### This code was initially implemented on a jupyter notebook



import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


appearances = pd.read_csv('appearances.csv')
club_games = pd.read_csv('club_games.csv')
clubs = pd.read_csv('clubs.csv')
game_events = pd.read_csv('game_events.csv')
game_lineups = pd.read_csv('game_lineups.csv')
games = pd.read_csv('games.csv')
player_valuations = pd.read_csv('player_valuations.csv')
players = pd.read_csv('players.csv')



### Creation of a dataframe with relevant player's data
appearances = appearances[['player_id', 'game_id', 'date', 'yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played']]
game_lineups = game_lineups[['game_id', 'player_id', 'type', 'position']]
player_valuations = player_valuations[['player_id', 'market_value_in_eur', 'current_club_id', 'player_club_domestic_competition_id']]
players = players[['player_id', 'last_season', 'country_of_citizenship', 'date_of_birth', 'position', 'highest_market_value_in_eur']]



### We agregate all appearances information for each player
appearances_aggregated = appearances.groupby('player_id').agg({
    'yellow_cards': 'sum',
    'red_cards': 'sum',
    'goals': 'sum',
    'assists': 'sum',
    'minutes_played': 'sum'
}).reset_index()



### The selected information is merged into a single dataframe
players_total = pd.merge(appearances_aggregated, player_valuations , on = 'player_id', how = 'outer')
players_total = pd.merge(players_total, players, on='player_id', how='outer')



### For each player, only the highest market value is retained
idx = players_total.groupby('player_id')['market_value_in_eur'].idxmax()

players_total = players_total.loc[idx]
players_total = players_total.drop('highest_market_value_in_eur', axis=1)
players_total['date_of_birth'] = pd.to_datetime(players_total['date_of_birth'])
players_total['log_market_value_in_eur'] = np.log(players_total['market_value_in_eur'] )



### Keeping only the numerical data for simplification
players_total = players_total[players_total.select_dtypes(include=[np.number]).columns]



### Split the data into train and test 
train, test = train_test_split(players_total, test_size=0.5, random_state=42)



### Save the new datasets in the current working directory (this code was run on a notebook)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)







