import pandas as pd

states = ['minnesota', 'california', 'pennsylvania', 'iowa', 'colorado', 'wisconsin', 'michigan', 'ohio', 'texas', 'washington', 'utah']

lines = []

with open('outbreak_data/uncleaned_outbreaks.txt', 'r') as file:
    file.readline()
    for line in file:
        split_line = line.strip().split(' ')
        date = split_line[0]
        cases = cases = int(split_line[-1].replace(',', ''))
        for s in states:
            if s in line.lower():
                lines.append({'Date': date, 'State': s.lower(), 'Cases': cases})
                break

data = pd.DataFrame(lines)

# print(data)

data.to_csv('csv_files/outbreaks.csv', index=False)

data['Week'] = pd.to_datetime(data['Date'], format='mixed').dt.to_period('W')

weekly_cases = data.groupby(['Week'])['Cases'].sum().reset_index()
weekly_cases.to_csv('csv_files/outbreaks_weekly.csv', index=False)

for state, group in data.groupby('State'):
    state_weekly_cases = group.groupby(['Week'])['Cases'].sum().reset_index()
    state_weekly_cases.to_csv(f'csv_files/outbreaks_by_state/{state}.csv', index=False)