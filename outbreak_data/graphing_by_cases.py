import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_bar_chart(data, file):

    data['Week'] = pd.to_datetime(data['Week'])
    data['Cases'] = data['Cases'] / 10**3
    
    all_weeks = pd.date_range(start=data['Week'].min(), end=data['Week'].max(), freq='4W')

    plt.figure(figsize=(20, 10))
    plt.plot(data['Week'], data['Cases'])
    plt.xlabel('Date (Year-week)')
    plt.ylabel('Cases (Thousands)')
    # Ensure all weeks are displayed
    plt.xticks(all_weeks, rotation=45)
    plt.gca().set_xticks(all_weeks)
    plt.gca().set_xticklabels([date.strftime('%Y-%m-%d') for date in all_weeks])

    # Save the plot
    plt.savefig(file)
    plt.close()

# Plot for each state
folder = 'csv_files/outbreaks_cases_by_state'
files = os.listdir(folder)

for file in files:
    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]
    plot_bar_chart(data, f'graphs/outbreaks_cases_weekly_by_state/{state}.png')

# Plot for the combined data
data = pd.read_csv('csv_files/outbreaks_cases_weekly.csv')
plot_bar_chart(data, 'graphs/outbreaks_cases_weekly_graph.png')