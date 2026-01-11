#game_id = '0021500012'

from event_info import event_dict
from event_info import event_desc

import json
import requests
import numpy as np

INITIAL_ARRAY_SIZE = 1000

def get_play_by_play(game_id, load_from_file=False, save_to_file=False):

    if load_from_file:
        pbp_json = json.load(open('{}_pbp.json'.format(game_id), 'r'))
    else:
        pbp_url = "http://stats.nba.com/stats/playbyplayv2?EndPeriod=10&EndRange=55800&GameID={}&RangeType=2&Season=2016-17&SeasonType=Regular+Season&StartPeriod=1&StartRange=0".format(str(game_id))
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                                 '(KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'}
        response = requests.get(pbp_url, headers=headers)
        response.raise_for_status()
        pbp_json = response.json()

    if save_to_file:
        with open('{}_pbp.json'.format(game_id), 'w') as outfile:
            json.dump(pbp_json, outfile)


    play_by_play = pbp_json['resultSets'][0]['rowSet']

    col_headers = ['msg_type', 'action_type',
                   'shot_made', 'shot_missed', 'points',
                   'assist', 'rebound', 'turnover1', 'turnover2']

    #pbp_array = np.zeros((INITIAL_ARRAY_SIZE, len(col_headers)))
    pbp_dict = {}

    # Monitor the score change on each event to look for 2 vs 3 pointers
    score = 0.0
    score_delta = 0.0

    for play in play_by_play:
        pbp_array = np.zeros((1, len(col_headers)))

        msg_type = play[2]
        action_type = play[3]

        # Extract clock and score data from the raw data
        quarter = play[4]
        raw_clock = play[6].split(':')
        game_clock = (4.0 - float(quarter)) * 12 * 60 + \
                      60.0 * float(raw_clock[0]) + float(raw_clock[1])

        if play[10]:
            raw_score = play[10].split(' - ')
            new_score = float(raw_score[0]) + float(raw_score[1])
            score_delta = new_score - score
            score = new_score

        # Use the description strings to look for assists
        descH = str(play[7])
        descN = str(play[8])
        descV = str(play[9])

        # When we have events, tag with the player id
        p1_id = play[13]
        p2_id = play[20]
        p3_id = play[27]

        pbp_array[0,0] = float(msg_type)
        pbp_array[0,1] = float(action_type)

        try:
            e = event_dict[msg_type]
            ed = event_desc[e][action_type]

            if e == 'rebound':
                pbp_array[0,6] = float(p1_id)

            elif e == 'turnover':
                if p1_id:
                    pbp_array[0,7] = float(p1_id)
                if p2_id:
                    pbp_array[0,8] = float(p2_id)

            elif e == 'shot_made':
                pbp_array[0,2] = float(p1_id)
                pbp_array[0,4] = score_delta

            elif e == 'shot_missed':
                pbp_array[0,3] = float(p1_id)
                pbp_array[0,4] = score_delta

            if 'AST' in descH or 'AST' in descV or 'AST' in descN:
                pbp_array[0,5] = float(p2_id)

        except KeyError:
            print('action type not found: ', msg_type, '-', action_type)
            if descH:
                print('HOME: ', descH)
            if descN:
                print('NEUT: ', descN)
            if descV:
                print('VIS: ', descV)
            print('')

        pbp_dict[game_clock] = pbp_array

    # Done entering data, trim rows higher than the index used
    return pbp_dict, col_headers

if __name__ == '__main__':
    game_id = '0021500001'
    data, headers = get_play_by_play(game_id, load_from_file=False, save_to_file=True)
    print(data)
