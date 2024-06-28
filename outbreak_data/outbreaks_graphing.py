import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

data = pd.read_csv('csv_files/outbreaks_weekly.csv')

data[['StartWeek', 'EndWeek']] = data['Week'].str.split('/', expand=True)
data['StartWeek'] = pd.to_datetime(data['StartWeek'])
data['Cases'] = data['Cases'] / 10**3

plt.figure(figsize=(10, 6))
plt.plot(data['StartWeek'], data['Cases'])
plt.xlabel('Week')
plt.ylabel('Cases (Thousands)')
# plt.show()
plt.savefig('graphs/oubreaks_weekly_graph.png')