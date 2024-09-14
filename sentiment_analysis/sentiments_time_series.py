import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_time_series(data, state):

    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data_grouped = data.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
    data_grouped.index = pd.to_datetime(data_grouped.index)

    weekly_data = data_grouped.resample('W').sum()  # Group by week

    # Ensure all weeks are included in the x-axis
    all_weeks = pd.date_range(
        start=weekly_data.index.min(), 
        end=weekly_data.index.max(), 
        freq='4W' 
    )

    print(all_weeks)

    plt.figure(figsize=(20, 10))
    plt.xticks(all_weeks, rotation=45)
    plt.gca().set_xticks(all_weeks)
    plt.gca().set_xticklabels([date.strftime('%Y-%m-%d') for date in all_weeks])

    for sentiment in weekly_data.columns:
        plt.plot(weekly_data.index, weekly_data[sentiment], label=sentiment)

    plt.xlabel('Date')
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