import matplotlib.pyplot as plt
from collections import Counter
import json
import os

def make_bar_charts(folder, count=10):

    specific_folder = '/'.join(folder.split('/')[:-1])
    bar_charts_path = f'{specific_folder}/bar_charts_by_state'

    print(f'Creating bar charts for {folder}...')

    files = os.listdir(folder)

    for file in files:
        # Open each JSON file and load its data
        with open(f'{folder}/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = file.split('.')[0]
        
        text = ''
        for topic, words in data.items():
            text += ' '.join(words)

        filtered_words = []
        for word in text.split():
            # Remove common terms used in data collection
            if word.lower() not in ['bird', 'birds', 'avian', 'flu', 'influenza', 'h5n1']:
                filtered_words.append(word)

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Get the most common 'top_n' words
        most_common_words = word_counts.most_common(count)

        # Separate the words (labels) and their counts
        labels, values = zip(*most_common_words)

        plt.figure(figsize=(17, 15))
        plt.rcParams.update({'font.size': 25})
        plt.bar(labels, values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Words")
        plt.ylabel("Frequency")

        plt.savefig(f'{bar_charts_path}/{state}.png')
        plt.close()

make_bar_charts('topic_modeling/BERTopic_method/topics_by_state')
make_bar_charts('topic_modeling/LDA_method/topics_by_state')