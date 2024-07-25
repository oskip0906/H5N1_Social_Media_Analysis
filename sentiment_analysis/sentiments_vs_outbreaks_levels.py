import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def generate_correlation_graph(sentiment):

    sentiments_data = pd.read_csv('csv_files/classified_comments.csv')
    cases_data = pd.read_csv('csv_files/outbreaks_monthly.csv')

    sentiments_data['Date'] = pd.to_datetime(sentiments_data['Date']).dt.date
    data_grouped = sentiments_data.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
    data_grouped.index = pd.to_datetime(data_grouped.index)

    sentiments_data = data_grouped.resample('ME').sum()
    min_val = sentiments_data[sentiment].min()
    max_val = sentiments_data[sentiment].max()
    sentiments_data[sentiment] = (sentiments_data[sentiment] - min_val) / (max_val - min_val) * 100

    cases_data['Month'] = pd.to_datetime(cases_data['Month'], format='%Y-%m')
    min_val_cases = cases_data['Cases'].min()
    max_val_cases = cases_data['Cases'].max()
    cases_data['Cases'] = (cases_data['Cases'] - min_val_cases) / (max_val_cases - min_val_cases) * 100

    cases_data.set_index('Month', inplace=True)

    # print(sentiments_data)
    # print(cases_data)

    # Calculate Pearson correlation
    sentiment_values = []
    cases_values = []

    for date in sentiments_data.index:
        month = date.strftime('%Y-%m')
        for month in cases_data.index:
            if date.strftime('%Y-%m') == month.strftime('%Y-%m'):
                sentiment_values.append(sentiments_data.loc[date, sentiment])
                cases_values.append(cases_data.loc[month,   'Cases'])

    correlation, _ = pearsonr(sentiment_values, cases_values)

    # Plotting the data in a line graph
    plt.figure(figsize=(14, 7))

    # Scale the data to be between 0 and 100
    plt.plot(sentiments_data.index, sentiments_data[sentiment], label=f'{sentiment} Level')
    plt.plot(cases_data.index, cases_data['Cases'], label='Cases')

    plt.xlabel('Date (Year - Month)')
    plt.ylabel('Normalized Value (0-100)')
    plt.legend()
    plt.text(0.5, 1.05, f"Pearson correlation coefficient: {correlation.round(3)}", ha='center', transform=plt.gca().transAxes)
    plt.savefig(f"graphs/outbreaks_vs_{sentiment}_levels.png")

generate_correlation_graph('Fear')
generate_correlation_graph('Anger')
generate_correlation_graph('Sadness')
generate_correlation_graph('Joy')