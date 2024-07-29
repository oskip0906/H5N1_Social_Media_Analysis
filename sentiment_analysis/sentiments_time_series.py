import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_time_series(data, state):

    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data_grouped = data.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
    data_grouped.index = pd.to_datetime(data_grouped.index)

    monthly_data = data_grouped.resample('ME').sum()

    # Ensure all months are included in the x-axis
    all_months = pd.date_range(
        start=monthly_data.index.min(), 
        end=monthly_data.index.max(), 
        freq='MS'
    )

    plt.figure(figsize=(20, 10))
    plt.xticks(all_months, rotation=45)
    plt.gca().set_xticks(all_months)
    plt.gca().set_xticklabels([date.strftime('%Y-%m') for date in all_months])

    for sentiment in monthly_data.columns:
        plt.plot(monthly_data.index, monthly_data[sentiment], label=sentiment)

    plt.xlabel('Date (Year-Month)')
    plt.ylabel('Number of Posts')
    plt.legend(loc='upper right')
    if state: 
        plt.savefig(f'graphs/sentiments_time_series_by_state/{state}.png')
    else:
        plt.savefig('graphs/sentiments_time_series.png')

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:
    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]
    plot_time_series(data, state)

data = pd.read_csv('csv_files/classified_comments.csv')
plot_time_series(data, None)