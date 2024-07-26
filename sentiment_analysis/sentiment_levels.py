import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_scores(data, sentiment):

    data = data[data['Sentiment'] == sentiment]

    data['Date'] = pd.to_datetime(data['Date']).dt.to_period('M')

    # Group by month and calculate mean intensity and count of sentiments
    data_grouped = data.groupby(data['Date']).agg(
        sentiment_count=('Sentiment', 'size'), 
        intensity=('Intensity', 'mean')
    ).fillna(0)

    data_grouped.index = data_grouped.index.to_timestamp()
    data = data_grouped.resample('ME').sum().fillna(0)

    # Comprehensive score 
    data['score'] = data['sentiment_count'] * data['intensity']

    # Normalize the sentiment intensity values
    min_val_score = data['score'].min()
    max_val_score = data['score'].max()
    data['score'] = (data['score'] - min_val_score) / (max_val_score - min_val_score) * 100

    return data

def plot_sentiment_levels_graphs(data, state):
    
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    data_grouped = data.groupby(['Date', 'Sentiment']).agg(
        intensity=('Intensity', 'mean')
    ).unstack(fill_value=0)

    data_grouped.index = pd.to_datetime(data_grouped.index)

    # Generate scores for each sentiment
    scores_dict = {}
    
    for sentiment in data_grouped.columns.levels[1]:
        sentiment_data = data_grouped.xs(sentiment, level=1, axis=1)
        sentiment_data = sentiment_data.reset_index()
        scores = generate_scores(data, sentiment)
        scores_dict[sentiment] = scores['score']

    plt.figure(figsize=(20, 10))
    
    # Ensure all months are included in the x-axis
    all_months = pd.date_range(start=scores.index.min(), end=scores.index.max(), freq='MS')
    plt.xticks(all_months, rotation=45)
    plt.gca().set_xticks(all_months)
    plt.gca().set_xticklabels([date.strftime('%Y-%m') for date in all_months])

    # Plot only the scores
    for sentiment, scores in scores_dict.items():
        plt.plot(scores.index, scores, label=f'{sentiment} Score')

    plt.xlabel('Date (Year-Month)')
    plt.ylabel('Normalized Score')
    plt.legend()
    if state: 
        plt.savefig(f'graphs/adjusted_sentiments_time_series_by_state/{state}.png')
    else:
        plt.savefig('graphs/adjusted_sentiments_time_series.png')
    
if __name__ == '__main__':

    folder = 'csv_files/classified_comments_by_state'
    files = os.listdir(folder)

    for file in files:
        data = pd.read_csv(f'{folder}/{file}')
        state = file[:-4].split('/')[-1]
        plot_sentiment_levels_graphs(data, state)
    
    data = pd.read_csv('csv_files/classified_comments.csv')
    plot_sentiment_levels_graphs(data, None)