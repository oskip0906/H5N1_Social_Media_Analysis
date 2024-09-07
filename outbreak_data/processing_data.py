import pandas as pd

states = ['minnesota', 'california', 'pennsylvania', 'iowa', 'colorado', 'wisconsin', 'michigan', 'ohio', 'texas', 'washington', 'utah']

lines = []

with open('outbreak_data/uncleaned_outbreaks.txt', 'r') as file:
    file.readline()
    for line in file:
        split_line = line.strip().split(' ')
        date = split_line[0]
        cases = int(split_line[-1].replace(',', ''))
        for s in states:
            if s in line.lower():
                lines.append({'Date': date, 'State': s.lower(), 'Cases': cases})
                break

data = pd.DataFrame(lines)

data.to_csv('csv_files/outbreaks.csv', index=False)

# Ensure the Date column is of datetime type
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data['Week'] = data['Date'].dt.to_period('W')

# Create a date range covering all possible weeks in the data
complete_week_range = pd.period_range(start=data['Week'].min(), end=data['Week'].max())

# Group by week and cases, including weeks with 0 cases
weekly_cases = data.groupby(['Week'])['Cases'].sum().reindex(complete_week_range, fill_value=0).reset_index()
weekly_cases.columns = ['Week', 'Cases']
weekly_cases['Week'] = weekly_cases['Week'].dt.start_time
weekly_cases.to_csv('csv_files/outbreaks_cases_weekly.csv', index=False)

# Get the number of entries for each week
data['Entries'] = data['Week'].map(data['Week'].value_counts())
weekly_entries = data.groupby('Week').size().reindex(complete_week_range, fill_value=0).reset_index()
weekly_entries.columns = ['Week', 'Entries']
weekly_entries['Week'] = weekly_entries['Week'].dt.start_time
weekly_entries.to_csv('csv_files/outbreaks_entries_weekly.csv', index=False)

for state, group in data.groupby('State'):

    # Group by week and cases, including weeks with 0 cases
    state_weekly_cases = group.groupby(['Week'])['Cases'].sum().reindex(complete_week_range, fill_value=0).reset_index()
    state_weekly_cases.columns = ['Week', 'Cases']
    state_weekly_cases['Week'] = state_weekly_cases['Week'].dt.start_time
    state_weekly_cases.to_csv(f'csv_files/outbreaks_cases_by_state/{state}.csv', index=False)
    
    # Group by week and entries, including weeks with 0 entries
    state_weekly_entries = group.groupby('Week').size().reindex(complete_week_range, fill_value=0).reset_index()
    state_weekly_entries.columns = ['Week', 'Entries']
    state_weekly_entries['Week'] = state_weekly_entries['Week'].dt.start_time
    state_weekly_entries.to_csv(f'csv_files/outbreaks_entries_by_state/{state}.csv', index=False)