import requests
from datetime import datetime

# submissions_link = 'https://api.pullpush.io/reddit/search/submission/?'
comments_link = 'https://api.pullpush.io/reddit/search/comment/?'

parameters = {
    'q': 'H5N1',
    'sort': 'asc',
    'size': 100,
    'after': 1672531200
}

for key, value in parameters.items():
    # submissions_link += f'{key}={value}&'
    comments_link += f'{key}={value}&'

# print(submissions_link)
# print(comments_link)

# submission_response = requests.get(submissions_link)
comment_response = requests.get(comments_link)

# submissions = submission_response.json()
comments = comment_response.json()

# print(comments['data'][0].keys())

# print("\n\nSubmissions:\n\n")

# for submission in submissions['data']:
#     print(submission['title'])
#     print('------------------------------------')

print("\n\nComments:\n\n")

for comment in comments['data']:
    try:
        print(comment['body'])
        # print(comment['created_utc'])
        date = datetime.utcfromtimestamp(comment['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
        print(date)
        print('------------------------------------')
    except KeyError:
        continue;