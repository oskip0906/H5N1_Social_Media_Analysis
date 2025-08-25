import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    return ((series - lo) / (hi - lo) * 100) if hi != lo else series * 0

def load_overall(sentiment: str):
    sentiments_data = pd.read_csv('csv_files/classified_comments.csv',
                                  parse_dates=['Date'])
    if sentiment != 'Overall':
        sentiments_data = sentiments_data[sentiments_data['Sentiment'] == sentiment]

    sentiments_data = sentiments_data[sentiments_data['Date'] >= '2022-02-21']
    sentiments_data['Date'] = sentiments_data['Date'].dt.to_period('W')
    sentiments_data = (sentiments_data
                       .groupby('Date')
                       .agg(sentiment_count=('Sentiment', 'size'),
                            intensity=('Intensity', 'mean')))
    sentiments_data.index = sentiments_data.index.to_timestamp()
    sentiments_data = sentiments_data.resample('W-MON').sum().fillna(0)

    outbreaks_data = pd.read_csv('csv_files/outbreaks_cases_weekly.csv',
                                 parse_dates=['Week']).set_index('Week')

    # build score & normalise both series
    sentiments_data['score'] = normalize(
        sentiments_data['sentiment_count'] * sentiments_data['intensity'])
    outbreaks_data['Cases'] = normalize(outbreaks_data['Cases'])

    # Restrict to common weeks
    common_weeks = sentiments_data.index.intersection(outbreaks_data.index)
    return sentiments_data.loc[common_weeks], outbreaks_data.loc[common_weeks]

def drop_jan_2023(df: pd.DataFrame):
    """Remove 2023-01-01 .. 2023-01-31 rows."""
    return df[~((df.index >= '2023-01-01') & (df.index <= '2023-01-31'))]

def shift_3w(df: pd.DataFrame):
    """Lag sentiment 3 weeks (shift forward)."""
    return df.shift(-3).iloc[:-3]

def build_plots(sentiment='Overall'):
    sentiments_data, outbreaks_data = load_overall(sentiment)

    normal_sentiments_data = sentiments_data.copy()
    shifted_sentiments_data = shift_3w(sentiments_data.copy())
    no_outliers_sentiments_data = drop_jan_2023(sentiments_data.copy())
    no_outliers_shifted_sentiments_data = shift_3w(drop_jan_2023(sentiments_data.copy()))

    variants = {
        'Normal'                : normal_sentiments_data,
        'Shifted'               : shifted_sentiments_data,
        'No Outliers'           : no_outliers_sentiments_data,
        'No Outliers + Shifted' : no_outliers_shifted_sentiments_data
    }
    colours  = {'Normal':'blue', 'Shifted':'green',
                'No Outliers':'purple', 'No Outliers + Shifted':'orange'}
    linestyles = {'Normal':'-', 'Shifted':'--',
                  'No Outliers':':', 'No Outliers + Shifted':'-.'}

    # Layered comparison plot
    plt.figure(figsize=(20, 10))
    for lab, df in variants.items():
        weeks = df.index
        pcc, _ = pearsonr(df['score'], outbreaks_data.loc[weeks, 'Cases'])
        plt.plot(weeks, df['score'],
                 label=f'{lab} {sentiment} (r={pcc:.3f})',
                 color=colours[lab], linestyle=linestyles[lab])

    plt.plot(outbreaks_data.index, outbreaks_data['Cases'],
             label='Outbreaks Level', color='red')
    # highlight outlier month
    plt.axvspan(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31'),
                alpha=0.2, color='gray', label='Outlier Period')

    plt.xlabel('Date')
    plt.ylabel('Normalized Value for Level (0-100)')
    plt.legend(loc='upper right')
    os.makedirs('graphs/outbreaks_cases_vs_sentiment_levels/',
                exist_ok=True)
    plt.savefig(f'graphs/outbreaks_cases_vs_sentiment_levels/'
                f'outbreaks_vs_{sentiment}_comparison.png',
                facecolor='white')
    plt.close()

    # Individual variant plots
    for lab, df in variants.items():
        weeks = df.index
        pcc, _ = pearsonr(df['score'], outbreaks_data.loc[weeks, 'Cases'])

        plt.figure(figsize=(20, 10))
        plt.plot(weeks, df['score'],
                 label=f'{lab} {sentiment}',
                 color=colours[lab], linestyle=linestyles[lab])
        plt.plot(weeks, outbreaks_data.loc[weeks, 'Cases'],
                 label='Outbreaks Level', color='red')

        plt.xlabel('Date')
        plt.ylabel('Normalized Value for Level (0-100)')
        plt.title(f'Pearson Correlation Coefficient: {pcc:.3f}')
        plt.legend(loc='upper right')
        plt.axvspan(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31'),
                    alpha=0.2, color='gray', label='Outlier Period')

        fname = f'graphs/outbreaks_cases_vs_sentiment_levels/' \
                f'outbreaks_vs_{sentiment}_{lab.replace(" ", "_").lower()}.png'
        plt.savefig(fname, facecolor='white')
        plt.close()

if __name__ == '__main__':
    sentiments = ['Sadness', 'Fear', 'Anger', 'Joy', 'Overall']
    for s in sentiments:
        build_plots(s)
