import requests

url = "http://stats.nba.com/stats/locations_getmoments/?eventid=308&gameid;=0041400235"
url = 'http://stats.nba.com/movement/#!/?GameID=0041400235&GameEventID;=308'
      'http://stats.nba.com/movement/#!/?GameID=0041400235&GameEventID;=308'

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'}

response = requests.get(url, headers=headers)
response.raise_for_status()
pbp_json = response.json()
