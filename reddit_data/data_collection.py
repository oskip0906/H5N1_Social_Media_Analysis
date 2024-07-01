import requests
import pandas as pd
from datetime import datetime

link_structure = 'https://api.pullpush.io/reddit/search/comment/?'

param_structure = {
    'sort': 'asc',
    'size': 100,
    'after': int(datetime.strptime('2022-02-08', "%Y-%m-%d").timestamp())
}

bird_flu = {
    'q': '"bird%20flu"',
    'param_structure': param_structure
}

avian_flu = {
    'q': '"avian%20flu"',
    'param_structure': param_structure
}

avian_influenza = {
    'q': '"avian%20influenza"',
    'param_structure': param_structure
}

h5n1 = {
    'q': 'h5n1',
    'param_structure': param_structure
}

parameters = [bird_flu, avian_flu, avian_influenza, h5n1]

comments_links = []

for parameter in parameters:
    param_structure = parameter['param_structure']
    param_query = parameter['q']
    comments_links.append(f'{link_structure}q={param_query}&{"&".join(f"{key}={value}" for key, value in param_structure.items())}')

print(comments_links)

subreddits = ['minnesota', 'southdakota', 'california', 'pennsylvania', 'iowa', 'colorado', 'wisconsin', 
              'michigan', 'ohio', 'texas', 'washington', 'utah', 'maryland', 'kansas']

links = []

for subreddit in subreddits:
    for link in comments_links:
        links.append(f'{link}&subreddit={subreddit}')

# print(links)

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