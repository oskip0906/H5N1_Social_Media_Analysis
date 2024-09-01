import matplotlib.pyplot as plt
import pandas as pd
import os

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:

    data = pd.read_csv(f'{folder}/{file}')
    state = file[:-4].split('/')[-1]
    sorted_labels = data['Sentiment'].value_counts().to_dict()

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels.keys(), sorted_labels.values())
    plt.title(f'{state.capitalize()}')
    plt.xlabel('Sentiment Types')
    plt.ylabel('Number of Posts')
    # plt.show()
    plt.savefig(f'graphs/classified_comments_by_state/{state}.png')

data = pd.read_csv('csv_files/classified_comments.csv')

sorted_labels = data['Sentiment'].value_counts().to_dict()

plt.figure(figsize=(12, 6))
plt.bar(sorted_labels.keys(), sorted_labels.values())
plt.xlabel('Sentiment Types')
plt.ylabel('Number of Posts')
# plt.show()
plt.savefig('graphs/classified_comments_graph.png')