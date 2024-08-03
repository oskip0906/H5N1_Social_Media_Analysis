import matplotlib.pyplot as plt
import json
import os

def make_topics_sentences(folder):

    specific_folder = '/'.join(folder.split('/')[:-1])

    files = os.listdir(folder)
    data = {}

    for file in files:

        with open(f'{folder}/{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = file.split('.')[0]

        text = ''

        for topic, words in data.items():
            sentence = ' '.join(words)
            text += ' '.join(set(sentence.split(' ')))
            text += '\n'

        if folder.split('/')[2] == 'negative_topics_by_state':
            with open(f'{specific_folder}/negative_topic_sentences_by_state/{state}.txt', 'w') as f:
                f.write(text)
        else:
            with open(f'{specific_folder}/topic_sentences_by_state/{state}.txt', 'w') as f:
                f.write(text)
        
make_topics_sentences('topic_modeling/BERTopic_method/topics_by_state')
make_topics_sentences('topic_modeling/LDA_method/topics_by_state')
make_topics_sentences('topic_modeling/BERTopic_method/negative_topics_by_state')
make_topics_sentences('topic_modeling/LDA_method/negative_topics_by_state')