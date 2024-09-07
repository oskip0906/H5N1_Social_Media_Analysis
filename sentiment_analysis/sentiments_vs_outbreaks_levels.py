import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_correlation_graph(sentiment, shifted=False):

    sentiments_data = pd.read_csv('csv_files/classified_comments.csv')
    outbreaks_data = pd.read_csv(f'csv_files/outbreaks_cases_weekly.csv')

    if sentiment == 'Overall':
        sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Neutral', 'Surprise', 'Love']
    else:
        sentiments = [sentiment]

    sentiments_data = sentiments_data[sentiments_data['Sentiment'].isin(sentiments)]

    # Convert the date to datetime and group by week, start from 2022-02-21
    sentiments_data['Date'] = pd.to_datetime(sentiments_data['Date'])
    sentiments_data = sentiments_data[sentiments_data['Date'] >= pd.Timestamp('2022-02-21')]
    sentiments_data['Date'] = sentiments_data['Date'].dt.to_period('W')

    # Group by week and calculate overall intensity and count of sentiments
    sentiments_data = sentiments_data.groupby(sentiments_data['Date']).agg(
        sentiment_count=('Sentiment', 'size'), 
        intensity=('Intensity', 'mean')
    ).fillna(0)

    sentiments_data.index = sentiments_data.index.to_timestamp()
    sentiments_data = sentiments_data.resample('W-MON').sum().fillna(0)

    # Comphrehensive score 
    sentiments_data['score'] = sentiments_data['sentiment_count'] * sentiments_data['intensity']

    # Normalize the sentiment intensity values
    min_score = sentiments_data['score'].min()
    max_score = sentiments_data['score'].max()
    sentiments_data['score'] = (sentiments_data['score'] - min_score) / (max_score - min_score) * 100

    # Normalize the cases values
    outbreaks_data['Week'] = pd.to_datetime(outbreaks_data['Week'], format='%Y-%m')
    min_val = outbreaks_data['Cases'].min()
    max_val = outbreaks_data['Cases'].max()
    outbreaks_data['Cases'] = (outbreaks_data['Cases'] - min_val) / (max_val - min_val) * 100

    outbreaks_data.set_index('Week', inplace=True)

    # Shift the sentiments data back by one week for exploration
    if shifted:
        sentiments_data = sentiments_data.shift(-3)
        sentiments_data = sentiments_data.iloc[:-3]

    # Calculate Pearson correlation
    sentiment_levels = []
    cases_levels = []
    weeks = []

    for date1 in sentiments_data.index:
        week1 = date1.strftime('%Y-%m-%d')
        for date2 in outbreaks_data.index:
            week2 = date2.strftime('%Y-%m-%d')
            if week1 == week2:
                sentiment_levels.append(sentiments_data.loc[date1, 'score'])
                cases_levels.append(outbreaks_data.loc[date2, 'Cases'])
                weeks.append(pd.to_datetime(week1))

    correlation, _ = pearsonr(sentiment_levels, cases_levels)

    # Plotting the data in a line graph
    plt.figure(figsize=(20, 10))
    plt.xticks(weeks, rotation=45)
    plt.gca().set_xticks(weeks)
    plt.gca().set_xticks(weeks[::4])
    plt.gca().set_xticklabels([date.strftime('%Y-%m-%d') for date in weeks[::4]])
    plt.plot(weeks, sentiment_levels, label=f'{sentiment} Level')
    plt.plot(weeks, cases_levels, label='Outbreaks Level')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value for Level (0-100)')
    plt.legend(loc='upper right')
    plt.text(0.5, 1.05, f"Pearson correlation coefficient: {correlation.round(3)}", ha='center', transform=plt.gca().transAxes)

    if shifted:
        plt.savefig(f"graphs/outbreaks_cases_vs_sentiment_levels/shifted_outbreaks_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")
    else:
        plt.savefig(f"graphs/outbreaks_cases_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")

# Generate correlation graphs for main sentiments
sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Overall']

for sentiment in sentiments:
    generate_correlation_graph(sentiment)
    generate_correlation_graph(sentiment, shifted=True)