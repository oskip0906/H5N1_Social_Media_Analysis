from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os

folder = 'json_files'
files = os.listdir(folder)

for file in files:

    with open(f'{folder}/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)

    state = file.split('.')[0]

    for topic, words in data.items():
        
        text = ' '.join(words)

        wordcloud = WordCloud().generate(text)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear') 
        plt.axis('off')
        plt.savefig(f'graphs/word_clouds_by_state/{state}.png')
        plt.close()