import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re  

data = pd.read_csv('csv_files/classified_comments.csv')
comments = data['Comment'].tolist()

print(comments)

processed_comments = []

for comment in comments:
    comment = re.sub(r'[^\w\s]', '', comment)
    processed_comments.append(comment)

text = ' '.join(processed_comments)

print(text)

wordcloud = WordCloud().generate(text)

plt.figure(figsize=(12, 8));
plt.imshow(wordcloud)
plt.axis('off')
# plt.show()
plt.savefig('graphs/word_cloud_graph.png')