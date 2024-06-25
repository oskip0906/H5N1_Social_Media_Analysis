import pandas as pd

data = pd.read_csv('csv_files/outbreaks.csv')

data['Week'] = pd.to_datetime(data['Outbreak Date'])
data.set_index('Week', inplace=True)

weekly_counts = data.resample('W').size()
weekly_counts_df = weekly_counts.to_frame(name='Count')

weekly_counts_df.to_csv('csv_files/outbreaks_weekly.csv')