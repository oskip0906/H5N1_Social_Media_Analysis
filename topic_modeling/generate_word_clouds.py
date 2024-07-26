from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os

def make_word_clouds(folder):

    specific_folder = '/'.join(folder.split('/')[:-1])

    files = os.listdir(folder)
    data = {}

    for file in files:

        with open(f'{folder}/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = file.split('.')[0]

        for topic, words in data.items():
            
            text = ' '.join(words)

            wordcloud = WordCloud(background_color="white", height=400, width=800).generate(text)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear') 
            plt.axis('off')
            plt.savefig(f'{specific_folder}/word_clouds_by_state/{state}.png')
            plt.close()

# make_word_clouds('topic_modeling/BERTopic_method/topics_by_state')
make_word_clouds('topic_modeling/LDA_method/topics_by_state')