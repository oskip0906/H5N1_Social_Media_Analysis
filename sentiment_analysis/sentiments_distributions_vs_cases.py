import pandas as pd
import matplotlib.pyplot as plt
import os

folder1 = 'csv_files/classified_comments_by_state'
files1 = os.listdir(folder1)

folder2 = 'csv_files/outbreaks_by_state'
files2 = os.listdir(folder2)

# (state, sentiment counts, cases)
state_data = []

for file in files1:
    data1 = pd.read_csv(f'{folder1}/{file}')
    data2 = pd.read_csv(f'{folder2}/{file}')

    state = file[:-4].split('/')[-1]
    sentiment_count = data1['Sentiment'].value_counts().to_dict()
    cases = data2['Cases'].sum()

    # Sum up all negative sentiment counts
    sentiment_count['Negative'] = sentiment_count.get('Sadness', 0) + sentiment_count.get('Anger', 0) + sentiment_count.get('Fear', 0)
    print(state, sentiment_count, cases)

    state_data.append((state, sentiment_count, cases))

# Sort by sentiment counts
sorted_by_sentiments = sorted(state_data, key=lambda x: x[1].get('Negative'))
states_sorted_by_sentiments = [state for state, _, _ in sorted_by_sentiments]

# Sort by cases
sorted_by_cases = sorted(state_data, key=lambda x: x[2])
states_sorted_by_cases = [state for state, _, _ in sorted_by_cases]

print("States sorted by negative sentiments (ascending):", states_sorted_by_sentiments)
print("States sorted by cases (ascending):", states_sorted_by_cases)