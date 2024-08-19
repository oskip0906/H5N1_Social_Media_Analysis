import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_correlation_graph(sentiment):

    sentiments_data = pd.read_csv('csv_files/classified_comments.csv')
    cases_data = pd.read_csv('csv_files/outbreaks_monthly.csv')

    if sentiment == 'Overall':
        sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Neutral', 'Surprise', 'Love']
    else:
        sentiments = [sentiment]

    sentiments_data = sentiments_data[sentiments_data['Sentiment'].isin(sentiments)]

    sentiments_data['Date'] = pd.to_datetime(sentiments_data['Date']).dt.to_period('M')

    # Group by month and calculate mean intensity and count of sentiments
    data_grouped = sentiments_data.groupby(sentiments_data['Date']).agg(
        sentiment_count=('Sentiment', 'size'), 
        intensity=('Intensity', 'mean')
    ).fillna(0)

    data_grouped.index = data_grouped.index.to_timestamp()
    sentiments_data = data_grouped.resample('M').sum().fillna(0)

    # Comphrehensive score 
    sentiments_data['score'] = sentiments_data['sentiment_count'] * sentiments_data['intensity']

    # Normalize the sentiment intensity values
    min_val_score = sentiments_data['score'].min()
    max_val_score = sentiments_data['score'].max()
    sentiments_data['score'] = (sentiments_data['score'] - min_val_score) / (max_val_score - min_val_score) * 100

    # Normalize the cases values
    cases_data['Month'] = pd.to_datetime(cases_data['Month'], format='%Y-%m')
    min_val_cases = cases_data['Cases'].min()
    max_val_cases = cases_data['Cases'].max()
    cases_data['Cases'] = (cases_data['Cases'] - min_val_cases) / (max_val_cases - min_val_cases) * 100

    cases_data.set_index('Month', inplace=True)

    # Shift the sentiments data back by one month for exploration
    sentiments_data = sentiments_data.shift(-1, freq='M')

    # print(sentiments_data)
    # print(cases_data)

    # Calculate Pearson correlation
    sentiment_levels = []
    cases_levels = []
    months = []

    for date1 in sentiments_data.index:
        month1 = date1.strftime('%Y-%m')
        for date2 in cases_data.index:
            month2 = date2.strftime('%Y-%m')
            if month1 == month2:
                sentiment_levels.append(sentiments_data.loc[date1, 'score'])
                cases_levels.append(cases_data.loc[date2, 'Cases'])
                months.append(month1)

    correlation, _ = pearsonr(sentiment_levels, cases_levels)

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
    plt.plot(all_months, cases_levels, label='Cases')
    plt.xlabel('Date (Year - Month)')
    plt.ylabel('Normalized Value (0-100)')
    plt.legend(loc='upper right')
    plt.text(0.5, 1.05, f"Pearson correlation coefficient: {correlation.round(3)}", ha='center', transform=plt.gca().transAxes)
    # plt.savefig(f"graphs/outbreaks_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")
    plt.savefig(f"graphs/adjusted_outbreaks_vs_sentiment_levels/outbreaks_vs_{sentiment}_levels.png")

generate_correlation_graph('Sadness')
generate_correlation_graph('Fear')
generate_correlation_graph('Anger')
generate_correlation_graph('Joy')

# All sentiments combined into 1 graph
generate_correlation_graph('Overall')