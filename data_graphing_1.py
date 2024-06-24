import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('csv_files/classified_data.csv')

labels = {
    "informative": 0,
    "frustrated": 0,
    "afraid": 0,
    "skeptical": 0,
    "critical": 0,
    "supportive": 0,
    "concerned": 0,
    "unconcerned": 0,
    "optimistic": 0
}

for index, row in data.iterrows():
    labels[row['Type']] += 1

sorted_labels = {k: v for k, v in sorted(labels.items(), key=lambda e:e[1], reverse=True)}

plt.figure(figsize=(12, 6))
plt.bar(sorted_labels.keys(), sorted_labels.values())
plt.xlabel('Categories')
plt.ylabel('Number of Posts')
# plt.show()
plt.savefig('graphs/classified_data_graph.png')