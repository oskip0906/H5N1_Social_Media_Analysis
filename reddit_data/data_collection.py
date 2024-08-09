import requests
import pandas as pd
from datetime import datetime
import json

# Define subreddits for each state
with open('reddit_data/states_to_subreddits.json', 'r') as file:
    states_to_subreddits = json.load(file)

# Define link for the API
link_structure = 'https://api.pullpush.io/reddit/search/comment/?'

# Define parameters 
param_structure = {
    'sort': 'asc',
    'size': 100,
    'after': int(datetime.strptime('2022-02-08', "%Y-%m-%d").timestamp())
}

# Define search queries
parameters = [
    {
        'q': '"bird%20flu"',
        'param_structure': param_structure
    },
    {
        'q': '"avian%20flu"',
        'param_structure': param_structure
    },
    {
        'q': '"avian%20influenza"',
        'param_structure': param_structure
    },
    {
        'q': 'h5n1',
        'param_structure': param_structure
    },
]

comments_links = []

for parameter in parameters:
    param_structure = parameter['param_structure']
    param_query = parameter['q']
    comments_links.append(f'{link_structure}q={param_query}&{"&".join(f"{key}={value}" for key, value in param_structure.items())}')

all_subreddits = states_to_subreddits['minnesota_subreddits'] + states_to_subreddits['southdakota_subreddits'] + \
                 states_to_subreddits['northern_california_subreddits'] + states_to_subreddits['southern_california_subreddits'] + \
                 states_to_subreddits['central_california_subreddits'] + states_to_subreddits['pennsylvania_subreddits'] + \
                 states_to_subreddits['iowa_subreddits'] + states_to_subreddits['colorado_subreddits'] + \
                 states_to_subreddits['wisconsin_subreddits'] + states_to_subreddits['michigan_subreddits'] + \
                 states_to_subreddits['ohio_subreddits'] + states_to_subreddits['texas_subreddits'] + \
                 states_to_subreddits['washington_subreddits'] + states_to_subreddits['utah_subreddits'] + \
                 states_to_subreddits['maryland_subreddits'] + states_to_subreddits['kansas_subreddits']

links = []

for subreddit in all_subreddits:
    for link in comments_links:
        links.append(f'{link}&subreddit={subreddit}')

comments_data = []

for link in links:
    comment_response = requests.get(link)
    comments = comment_response.json()
    count = 0
    for comment in comments['data']:
        try:
            date = datetime.fromtimestamp(comment['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
            subreddit = link.split('&')[-1].split('=')[-1]
            comments_data.append({'Comment': comment['body'], 'Date': date, 'Subreddit': subreddit})
            count += 1
        except Exception as e:
            print(f"error processing comment")
            continue
    print(count)

df = pd.DataFrame(comments_data)

print(df)

new_df = df.drop_duplicates(subset=['Comment'])
new_df.to_csv('csv_files/comments.csv', index=False)