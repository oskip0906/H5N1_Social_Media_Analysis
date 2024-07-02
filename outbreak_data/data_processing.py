import csv
import pandas as pd

states = ['minnesota', 'california', 'colorado', 'iowa', 'maryland', 'michigan', 'ohio', 'pennsylvania', 'texas', 'utah', 'wisconsin']
lines = []

with open('outbreak_data/uncleaned_outbreaks.txt', 'r') as file:
    file.readline()
    for line in file:
        split_line = line.strip().split(' ')
        date = split_line[0]
        cases = cases = int(split_line[-1].replace(',', ''))
        state = split_line[1]
        if state.lower() in states:
            lines.append({'Date': date, 'State': state.lower(), 'Cases': cases})

data = pd.DataFrame(lines)

print(data)

data.to_csv('csv_files/outbreaks.csv', index=False)

for state, state_df in data.groupby('State'):
    state_df.to_csv(f'csv_files/outbreaks_by_state/{state}.csv', index=False)

data['Week'] = pd.to_datetime(data['Date'], format='mixed').dt.to_period('W')

week_to_cases = data.groupby('Week')['Cases'].sum().to_dict()

# print(week_to_cases)

weekly_counts = pd.DataFrame.from_dict(week_to_cases, orient='index', columns=['Cases'])
weekly_counts.reset_index(inplace=True)

weekly_counts.columns = ['Week', 'Cases']

weekly_counts.to_csv('csv_files/outbreaks_weekly.csv', index=False)