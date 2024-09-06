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

data.to_csv('csv_files/outbreaks.csv', index=False)

# Ensure the Date column is of datetime type
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data['Month'] = data['Date'].dt.to_period('M')

# Create a date range covering all possible months in the data
complete_month_range = pd.period_range(
    start=data['Month'].min(), 
    end=data['Month'].max(), 
    freq='M'
)

# Group by Month and cases, including months with 0 cases
monthly_cases = data.groupby(['Month'])['Cases'].sum().reindex(complete_month_range, fill_value=0).reset_index()
monthly_cases.columns = ['Month', 'Cases']

monthly_cases.to_csv('csv_files/outbreaks_cases_monthly.csv', index=False)

# Get the number of entries for each month
data['Entries'] = data['Month'].map(data['Month'].value_counts())
monthly_entries = data.groupby('Month').size().reindex(complete_month_range, fill_value=0).reset_index()
monthly_entries.columns = ['Month', 'Entries']

monthly_entries.to_csv('csv_files/outbreaks_entries_monthly.csv', index=False)

for state, group in data.groupby('State'):

    # Group by month and cases, including months with 0 cases
    state_monthly_cases = group.groupby(['Month'])['Cases'].sum().reindex(complete_month_range, fill_value=0).reset_index()
    state_monthly_cases.columns = ['Month', 'Cases']
    state_monthly_cases.to_csv(f'csv_files/outbreaks_cases_by_state/{state}.csv', index=False)
    
    # Group by month and entries, including months with 0 entries
    state_monthly_entries = group.groupby('Month').size().reindex(complete_month_range, fill_value=0).reset_index()
    state_monthly_entries.columns = ['Month', 'Entries']
    state_monthly_entries.to_csv(f'csv_files/outbreaks_entries_by_state/{state}.csv', index=False)