import os
import pandas as pd
import matplotlib.pyplot as plt

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:

    if file == 'excluded_states.csv':
        continue

    data = pd.read_csv(f'{folder}/{file}')
    state = file.split('.')[0]

    # Distributions of topics by frequency in comments
    topic_counts = data[data['corresponding_topic'] != 'No topic']['corresponding_topic'].value_counts().to_dict()

    plt.figure(figsize=(20, 10))
    plt.bar(topic_counts.keys(), topic_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Corresponding Topic')
    plt.ylabel('Frequency of Topic in Comments')

    plt.savefig(f'graphs/topic_distributions_by_frequency_by_state/{state}.png')

    # Distributions of topics by intensity of negative comments

    negative_sentiments = ['Fear', 'Anger', 'Sadness']
    filtered_data = data[data['Sentiment'].isin(negative_sentiments)]

    topic_counts = filtered_data[filtered_data['corresponding_topic'] != 'No topic']['corresponding_topic'].value_counts().to_dict()
    intensity_sums = filtered_data.groupby('corresponding_topic')['Intensity'].sum().to_dict()

    combined_counts = {topic: topic_counts.get(topic) * intensity_sums.get(topic) for topic in topic_counts.keys()}

    plt.figure(figsize=(20, 10))
    plt.bar(combined_counts.keys(), combined_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Corresponding Topic')
    plt.ylabel('Aggregated Intensity of Negativivity in Comments')

    plt.savefig(f'graphs/topic_distributions_by_negative_sentiments_by_state/{state}.png')