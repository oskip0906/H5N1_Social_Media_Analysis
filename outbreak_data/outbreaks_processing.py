import pandas as pd

data = pd.read_csv('csv_files/outbreaks.csv', encoding='utf-16')

data['Week'] = pd.to_datetime(data['Date'], format='mixed').dt.to_period('W')

week_to_cases = data.groupby('Week')['BirdsAffected'].sum().to_dict()

# print(week_to_cases)

weekly_counts = pd.DataFrame.from_dict(week_to_cases, orient='index', columns=['Cases'])
weekly_counts.reset_index(inplace=True)

weekly_counts.columns = ['Week', 'Cases']

weekly_counts.to_csv('csv_files/outbreaks_weekly.csv', index=False)