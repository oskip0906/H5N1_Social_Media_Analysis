import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

data = pd.read_csv('csv_files/outbreaks_weekly.csv')

data['Week'] = pd.to_datetime(data['Week'])

plt.figure(figsize=(10, 6))
plt.plot(data['Week'], data['Count'])
plt.xlabel('Week')
plt.ylabel('Count')
# plt.show()
plt.savefig('graphs/oubreaks_weekly_graph.png')