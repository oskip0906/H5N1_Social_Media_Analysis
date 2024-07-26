import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_correlation_graph(sentiment):

    sentiments_data = pd.read_csv('csv_files/classified_comments.csv')
    cases_data = pd.read_csv('csv_files/outbreaks_monthly.csv')

    sentiments_data = sentiments_data[sentiments_data['Sentiment'] == sentiment]

    sentiments_data['Date'] = pd.to_datetime(sentiments_data['Date']).dt.to_period('M')

    # Group by month and calculate mean intensity and count of sentiments
    data_grouped = sentiments_data.groupby(sentiments_data['Date']).agg(
        sentiment_count=('Sentiment', 'size'), 
        intensity=('Intensity', 'mean')
    ).fillna(0)

    data_grouped.index = data_grouped.index.to_timestamp()
    sentiments_data = data_grouped.resample('ME').sum().fillna(0)

    # Comphrehensive score 
    sentiments_data['score'] = sentiments_data['sentiment_count'] * sentiments_data['intensity']

    # Normalize the sentiment intensity values
    min_val_score = sentiments_data['score'].min()
    max_val_score = sentiments_data['score'].max()
    sentiments_data['score'] = (sentiments_data['score'] - min_val_score) / (max_val_score - min_val_score) * 100

    # Shift the scores data back by 1 month for interesting discoveries
    sentiments_data.index = sentiments_data.index - pd.DateOffset(months=1)

    # Normalize the cases values
    cases_data['Month'] = pd.to_datetime(cases_data['Month'], format='%Y-%m')
    min_val_cases = cases_data['Cases'].min()
    max_val_cases = cases_data['Cases'].max()
    cases_data['Cases'] = (cases_data['Cases'] - min_val_cases) / (max_val_cases - min_val_cases) * 100

    cases_data.set_index('Month', inplace=True)

    # print(sentiments_data)
    # print(cases_data)

    # Calculate Pearson correlation
    sentiment_levels = []
    cases_levels = []

    for date in sentiments_data.index:
        month = date.strftime('%Y-%m')
        for month in cases_data.index:
            if date.strftime('%Y-%m') == month.strftime('%Y-%m'):
                sentiment_levels.append(sentiments_data.loc[date, 'score'])
                cases_levels.append(cases_data.loc[month, 'Cases'])

    correlation, _ = pearsonr(sentiment_levels, cases_levels)

    # Plotting the data in a line graph
    plt.figure(figsize=(14, 7))

    plt.plot(sentiments_data.index, sentiments_data['score'], label=f'{sentiment} Level')
    plt.plot(cases_data.index, cases_data['Cases'], label='Cases')

    plt.xlabel('Date (Year - Month)')
    plt.ylabel('Normalized Value (0-100)')
    plt.legend()
    plt.text(0.5, 1.05, f"Pearson correlation coefficient: {correlation.round(3)}", ha='center', transform=plt.gca().transAxes)
    plt.savefig(f"graphs/outbreaks_vs_{sentiment}_levels_shifted.png")

generate_correlation_graph('Fear')
# generate_correlation_graph('Anger')
# generate_correlation_graph('Joy')