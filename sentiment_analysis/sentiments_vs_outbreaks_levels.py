import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_correlation_graph(sentiment, cases_or_entries, shifted=False):

    sentiments_data = pd.read_csv('csv_files/classified_comments.csv')
    outbreaks_data = pd.read_csv(f'csv_files/outbreaks_{cases_or_entries.lower()}_monthly.csv')

    if sentiment == 'Overall':
        sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Neutral', 'Surprise', 'Love']
    else:
        sentiments = [sentiment]

    sentiments_data = sentiments_data[sentiments_data['Sentiment'].isin(sentiments)]

    sentiments_data['Date'] = pd.to_datetime(sentiments_data['Date']).dt.to_period('M')

    # Group by month and calculate overall intensity and count of sentiments
    data_grouped = sentiments_data.groupby(sentiments_data['Date']).agg(
        sentiment_count=('Sentiment', 'size'), 
        intensity=('Intensity', 'mean')
    ).fillna(0)

    data_grouped.index = data_grouped.index.to_timestamp()
    sentiments_data = data_grouped.resample('M').sum().fillna(0)

    # Comphrehensive score 
    sentiments_data['score'] = sentiments_data['sentiment_count'] * sentiments_data['intensity']

    # Normalize the sentiment intensity values
    min_score = sentiments_data['score'].min()
    max_score = sentiments_data['score'].max()
    sentiments_data['score'] = (sentiments_data['score'] - min_score) / (max_score - min_score) * 100

    # Normalize the cases values
    outbreaks_data['Month'] = pd.to_datetime(outbreaks_data['Month'], format='%Y-%m')
    min_val = outbreaks_data[cases_or_entries].min()
    max_val = outbreaks_data[cases_or_entries].max()
    outbreaks_data[cases_or_entries] = (outbreaks_data[cases_or_entries] - min_val) / (max_val - min_val) * 100

    outbreaks_data.set_index('Month', inplace=True)

    # Shift the sentiments data back by one month for exploration
    if shifted:
        sentiments_data = sentiments_data.shift(-1, freq='M')
        sentiments_data = sentiments_data.iloc[:-1]

    # Calculate Pearson correlation
    sentiment_levels = []
    cases_or_entries_levels = []
    months = []

    for date1 in sentiments_data.index:
        month1 = date1.strftime('%Y-%m')
        for date2 in outbreaks_data.index:
            month2 = date2.strftime('%Y-%m')
            if month1 == month2:
                sentiment_levels.append(sentiments_data.loc[date1, 'score'])
                cases_or_entries_levels.append(outbreaks_data.loc[date2, cases_or_entries])
                months.append(month1)

    correlation, _ = pearsonr(sentiment_levels, cases_or_entries_levels)

    all_months = pd.date_range(
        start=months[0],
        end=months[-1],
        freq='MS'
    )

    # Plotting the data in a line graph
    plt.figure(figsize=(15, 8))
    plt.xticks(all_months, rotation=45)
    plt.gca().set_xticks(all_months)
    plt.gca().set_xticklabels([date.strftime('%Y-%m') for date in all_months])
    plt.plot(all_months, sentiment_levels, label=f'{sentiment} Level')
    plt.plot(all_months, cases_or_entries_levels, label='Outbreaks Level')
    plt.xlabel('Date (Year - Month)')
    plt.ylabel('Normalized Value for Level (0-100)')
    plt.legend(loc='upper right')
    plt.text(0.5, 1.05, f"Pearson correlation coefficient: {correlation.round(3)}", ha='center', transform=plt.gca().transAxes)

    if shifted:
        plt.savefig(f"graphs/outbreaks_{cases_or_entries.lower()}_vs_sentiment_levels/shifted_outbreaks_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")
    else:
        plt.savefig(f"graphs/outbreaks_{cases_or_entries.lower()}_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")

# Generate correlation graphs for main sentiments
sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Overall']
cases_or_entries = ['Cases', 'Entries']
shifted = [False, True]

for sentiment in sentiments:
    for type in cases_or_entries:
        generate_correlation_graph(sentiment, type)

# Generate shifted graphs for outbreaks cases vs sentiments levels
for sentiment in sentiments:
    generate_correlation_graph(sentiment, 'Cases', shifted=True)