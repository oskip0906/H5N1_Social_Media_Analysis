from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import hdbscan
import umap
import pandas as pd
import re
import json

nltk.download('punkt')

data = pd.read_csv('csv_files/classified_comments.csv')
comments = data['Comment'].tolist()

def preprocess_comment(comment):
    comment = re.sub(r'[^\w\s]', '', comment).lower()
    tokens = nltk.word_tokenize(comment)
    filtered_tokens = []
    for word in tokens:
        if word not in stopwords.words('english'):
            filtered_tokens.append(word)
    return ' '.join(filtered_tokens)

processed_comments = []

for comment in comments:
    processed_comments.append(preprocess_comment(comment))

# 3 to 5 words per subtopic
vectorizer_model = CountVectorizer(ngram_range=(3, 5), stop_words='english', min_df=10, max_df=0.1)

# umap model for dimensionality reduction
umap_model = umap.UMAP(n_neighbors=10, n_components=5, min_dist=0, metric='cosine')

# hdbscan model for clustering
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2, cluster_selection_epsilon=0.5)

# bertopic model with adjusted parameters
topic_model = BERTopic(vectorizer_model=vectorizer_model, umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)

# fit-transform the model
topics, probabilities = topic_model.fit_transform(processed_comments)

topics_list = {}

i = 0
while topic_model.get_topic(i) != False:
    subtopics = topic_model.get_topic(i)
    subtopics_list = []
    for subtopic in subtopics:
        subtopics_list.append(subtopic[0])
    topics_list[f"Topics Group {i + 1}"] = subtopics_list
    i += 1

# print(topics_list)

with open('json_files/topics.json', 'w') as f:
    json.dump(topics_list, f, indent=4)