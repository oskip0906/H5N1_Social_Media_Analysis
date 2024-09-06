import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_bar_chart(data, file):

    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
    data['Cases'] = data['Cases'] / 10**3

    # Ensure all months are included in the x-axis
    all_months = pd.date_range(start=data['Month'].min(), end=data['Month'].max(), freq='MS')

    plt.figure(figsize=(20, 10))
    plt.plot(data['Month'], data['Cases'])
    plt.xlabel('Date (Year-Month)')
    plt.ylabel('Cases (Thousands)')
    # Ensure all months are displayed
    plt.xticks(all_months, rotation=45)
    plt.gca().set_xticks(all_months)
    plt.gca().set_xticklabels([date.strftime('%Y-%m') for date in all_months])

    # Save the plot
    plt.savefig(file)
    plt.close()

# Plot for each state
folder = 'csv_files/outbreaks_cases_by_state'
files = os.listdir(folder)

for file in files:
    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]
    plot_bar_chart(data, f'graphs/outbreaks_cases_monthly_by_state/{state}.png')

# Plot for the combined data
data = pd.read_csv('csv_files/outbreaks_cases_monthly.csv')
plot_bar_chart(data, 'graphs/outbreaks_cases_monthly_graph.png')