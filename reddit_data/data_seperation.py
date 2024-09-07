import pandas as pd
import json

# Define subreddits for each state
with open('reddit_data/states_to_subreddits.json', 'r') as file:
    states_to_subreddits = json.load(file)

comments_df = pd.read_csv("csv_files/comments.csv")

print(len(comments_df))

# Updated mapping of subreddits to their respective states
subreddits_to_states = {
    **dict.fromkeys(states_to_subreddits['minnesota_subreddits'], 'minnesota'),
    **dict.fromkeys(states_to_subreddits['southdakota_subreddits'], 'south_dakota'),
    **dict.fromkeys(states_to_subreddits['northern_california_subreddits'] + 
                    states_to_subreddits['southern_california_subreddits'] + 
                    states_to_subreddits['central_california_subreddits'], 'california'),
    **dict.fromkeys(states_to_subreddits['pennsylvania_subreddits'], 'pennsylvania'),
    **dict.fromkeys(states_to_subreddits['iowa_subreddits'], 'iowa'),
    **dict.fromkeys(states_to_subreddits['colorado_subreddits'], 'colorado'),
    **dict.fromkeys(states_to_subreddits['wisconsin_subreddits'], 'wisconsin'),
    **dict.fromkeys(states_to_subreddits['michigan_subreddits'], 'michigan'),
    **dict.fromkeys(states_to_subreddits['ohio_subreddits'], 'ohio'),
    **dict.fromkeys(states_to_subreddits['texas_subreddits'], 'texas'),
    **dict.fromkeys(states_to_subreddits['washington_subreddits'], 'washington'),
    **dict.fromkeys(states_to_subreddits['utah_subreddits'], 'utah'),
    **dict.fromkeys(states_to_subreddits['maryland_subreddits'], 'maryland'),
    **dict.fromkeys(states_to_subreddits['kansas_subreddits'], 'kansas')
}

subreddits = comments_df['Subreddit'].value_counts().to_dict()

states_dfs = {state: pd.DataFrame() for state in set(subreddits_to_states.values())}

excluded_states_dfs = pd.DataFrame()

for subreddit in subreddits:
    state = subreddits_to_states.get(subreddit)
    if states_dfs[state].empty:
        states_dfs[state] = comments_df[comments_df['Subreddit'] == subreddit]
    else:
        states_dfs[state] = pd.concat([states_dfs[state], comments_df[comments_df['Subreddit'] == subreddit]])

for state, state_df in states_dfs.items():
    if state_df.shape[0] < 50:
        excluded_states_dfs = pd.concat([excluded_states_dfs, state_df])
    else:
        state_df.to_csv(f'csv_files/comments_by_state/{state}.csv', index=False)

excluded_states_dfs.to_csv('csv_files/comments_by_state/excluded_states.csv', index=False)