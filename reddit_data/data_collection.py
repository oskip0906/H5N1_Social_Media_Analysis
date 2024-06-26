import requests
import re
import csv
from datetime import datetime

comments_link = 'https://api.pullpush.io/reddit/search/comment/?'

parameters = {
    'q': 'H5N1',
    'sort': 'asc',
    'sort_type': 'created_utc',
    'size': 100,
    'after': int(datetime.strptime('2022-01-01', "%Y-%m-%d").timestamp())
}

for key, value in parameters.items():
    comments_link += f'{key}={value}&'

# 100 comments max per request

link_1 = f'{comments_link}subreddit=H5N1_AvianFlu'
link_2 = f'{comments_link}subreddit=epidemiology'
link_3 = f'{comments_link}subreddit=medicine'
link_4 = f'{comments_link}subreddit=science'
link_5 = f'{comments_link}subreddit=news'
link_6 = f'{comments_link}subreddit=worldnews'
link_7 = f'{comments_link}subreddit=Health'
link_8 = f'{comments_link}subreddit=OutOfTheLoop'

links = [link_1, link_2, link_3, link_4, link_5, link_6, link_7, link_8]

print(links)

with open('csv_files/comments.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Comment', 'Date'])
    for link in links:
        comment_response = requests.get(link)
        comments = comment_response.json()
        count = 0
        for comment in comments['data']:
            try:
                date = datetime.fromtimestamp(comment['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
                encoded_body = comment['body'].encode('latin-1', errors='replace')
                writer.writerow([encoded_body.decode('latin-1'), date])
                count += 1
            except:
                continue
        print(count)