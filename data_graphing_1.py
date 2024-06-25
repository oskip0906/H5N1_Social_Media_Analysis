import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('csv_files/classified_data.csv')

sorted_labels = data['Type'].value_counts().to_dict()

plt.figure(figsize=(12, 6))
plt.bar(sorted_labels.keys(), sorted_labels.values())
plt.xlabel('Categories')
plt.ylabel('Number of Posts')
# plt.show()
plt.savefig('graphs/classified_data_graph.png')