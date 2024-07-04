import json
from wordcloud import WordCloud

import matplotlib.pyplot as plt

with open('json_files/topics.json', 'r') as file:

    data = json.load(file)

    words_list = []

    for value in data.values():
        words_list.append(str(value))

    words_list = set(words_list)

    text = ' '.join(words_list)

    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud)
    plt.axis('off')
    # plt.show()
    plt.savefig('graphs/word_cloud_graph.png')