import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

data = pd.read_csv('csv_files/classified_data.csv')

data['Date'] = pd.to_datetime(data['Date']).dt.date
data_grouped = data.groupby(['Date', 'Type']).size().unstack(fill_value=0)

fig, axs = plt.subplots(len(data_grouped.columns), 1, figsize=(30, 10), sharex=True)

for i, column in enumerate(data_grouped.columns):
    axs[i].plot(data_grouped.index, data_grouped[column], label=column)
    axs[i].legend(title='Type', loc='upper left')

axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  

plt.xlabel('Date (Year-Month)')
fig.text(0.1, 0.5, 'Number of Posts', va='center', rotation='vertical')
# plt.show()
plt.savefig('graphs/time_series_graph.png')