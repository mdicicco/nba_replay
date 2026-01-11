import json
import time
import requests
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import play_by_play as pbp
INITIAL_ARRAY_SIZE = 225000 #222857
PLAYERS_PER_SIDE = 13


def create_data_frame(game_id):

    # Open the json file
    all_data = json.load(open('{}_svu.json'.format(game_id), 'r'))
    game_date = all_data['gamedate']
    game_id = all_data['gameid']

    print("GAME: ", game_id, ' occuring on ', game_date)

    # Get the play by play for this event
    pbp_data, pbp_headers = pbp.get_play_by_play(game_id, load_from_file=True)

    # Create an empty data set to fill
    #  constantly resizing will be slow so start with a big array and trim when finished
    col_headers = ['quarter', 'game_clock', 'game_seconds', 'shot_clock', 'event_id', 'moment_id']
    col_headers += ['ball_x', 'ball_y', 'ball_z']

    home_player_col = len(col_headers)
    for player in range(13):
        prefix = 'home_player_{}'.format(player+1)
        col_headers += [prefix+'_id', prefix+'_team', prefix+'_x', prefix+'_y', prefix+'_r']

    away_player_col = len(col_headers)
    for player in range(13):
        prefix = 'away_player_{}'.format(player+1)
        col_headers += [prefix+'_id', prefix+'_team', prefix+'_x', prefix+'_y', prefix+'_r']

    pbp_start_col = len(col_headers)
    col_headers += pbp_headers

    df = np.zeros((INITIAL_ARRAY_SIZE, len(col_headers)))
    shot_data = np.zeros((1000, len(col_headers)))
    wall_time = np.zeros((INITIAL_ARRAY_SIZE, 1))

    # Create data objects for the players, which will be parsed on the first event
    player_data_inserted = False
    player_map = {}
    vis_players = set()
    home_players = set()

    def insert_player_data(df, event, home_player_col, away_player_col):

        # Player data is repeated for every event in the file, but it is only necessary
        # to extract is once.
        if not player_data_inserted:
            home = event['home']
            h_abbr = home['abbreviation']
            h_id = home['teamid']

            for player in home['players']:
                pid = player.pop('playerid')
                vis_players.add(pid)
                player_map[pid] = player

                # Populate the right column in the data frame with this player's ID
                df[:, home_player_col] = np.ones(INITIAL_ARRAY_SIZE) * float(pid)
                df[:, home_player_col+1] = np.ones(INITIAL_ARRAY_SIZE) * float(h_id)
                player_map[pid]['column'] = home_player_col
                home_player_col += 5  # this will wrap around to the next player

            visitor = event['visitor']
            v_abbr = visitor['abbreviation']
            v_id = visitor['teamid']

            for player in visitor['players']:
                pid = player.pop('playerid')
                home_players.add(pid)
                player_map[pid] = player

                # Populate the right column in the data frame with this player's ID
                df[:, away_player_col] = np.ones(INITIAL_ARRAY_SIZE) * float(pid)
                df[:, away_player_col+1] = np.ones(INITIAL_ARRAY_SIZE) * float(v_id)
                player_map[pid]['column'] = away_player_col
                away_player_col += 5  # this will wrap around to the next player

    # Extract the event list, and then iterate through it to parse events
    events_list = all_data['events']

    index = 0
    pbp_index = 0

    for event in events_list:
        event_id = int(event['eventId'])

        # Insert the player data once, it is provided in every frame
        if not player_data_inserted:
            insert_player_data(df, event, home_player_col, away_player_col)
            player_data_inserted = True

        # Extract all the data from the moments associated with this event
        moments_list = event['moments']
        for m_id, moment in enumerate(moments_list):

            wall_time[index] = float(str(moment[1])[0:10]) + \
                               float("0." + str(moment[1])[10:])

            df[index][0] = quarter = float(moment[0])  # quarter
            df[index][1] = game_clock = float(moment[2])  # game clock
            df[index][2] = seconds_left = (4.0 - float(quarter)) * 12.0 * 60.0 + game_clock  # seconds left

            if moment[3]:
                df[index][3] = float(moment[3])
            else:
                df[index][3] = -1.0

            UNKNOWN = moment[4]  # always seems to be None. Maybe OT?

            df[index][4] = float(event_id)
            df[index][5] = float(m_id)

            # Position is a list of lists containing the ball, and then all the players
            position_list = moment[5]
            _, _, b_x, b_y, b_z = position_list[0]
            df[index][6] = b_x
            df[index][7] = b_y
            df[index][8] = b_z

            # Find the player's spot, then put the data in the right columns
            for player in position_list[1:]:
                teamid, pid, x, y, r = player
                player_column = player_map[pid]['column']
                df[index][player_column] = float(pid)
                df[index][player_column+1] = float(teamid)
                df[index][player_column+2] = float(x)
                df[index][player_column+3] = float(y)
                df[index][player_column+4] = float(r)

            # If game clock and next pbp event match, insert pbp data
            if seconds_left in pbp_data:
                df[index][pbp_start_col:] = pbp_data[seconds_left]
                pbp_data.pop(seconds_left)

                shot_data[pbp_index] = df[index]
                pbp_index += 1

            index += 1

    #data = pd.DataFrame(data=df, index=wall_time, columns=col_headers)
    #print(data.head())
    return wall_time, df, index, shot_data, pbp_index

if __name__ == '__main__':
    game_id = '0021500001'
    index, data, count, shot_data, pbp_index = create_data_frame(game_id)
    print(data)

    np.savetxt("output_test2.csv", data, delimiter=",")
