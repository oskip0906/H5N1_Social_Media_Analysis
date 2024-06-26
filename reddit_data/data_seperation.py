import pandas as pd

comments_df = pd.read_csv("csv_files/comments.csv")

subreddits = comments_df['Subreddit'].value_counts().to_dict()
new_subreddits = {k: v for k, v in subreddits.items() if v >= 10}

subreddit_dfs = {}
for subreddit in new_subreddits:
    subreddit_dfs[subreddit] = comments_df[comments_df['Subreddit'] == subreddit]

# print(subreddit_dfs)

for subreddit, subreddit_df in subreddit_dfs.items():
    subreddit_df.to_csv(f'csv_files/comments_by_state/{subreddit}.csv', index=False)