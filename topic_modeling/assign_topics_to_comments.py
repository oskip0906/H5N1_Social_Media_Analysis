import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def state_comments_to_topics(data, state):

    with open(f'topic_modeling/BERTopic_method/comments_by_state/{state}.json', 'r') as f:
        topics1 = json.load(f)
        topics1 = {f'BERTopic {k}': v for k, v in topics1.items()}

    with open(f'topic_modeling/LDA_method/comments_by_state/{state}.json', 'r') as f:
        topics2 = json.load(f)
        topics2 = {f'LDA {k}': v for k, v in topics2.items()}

    topics = {**topics1, **topics2}

    data['corresponding_topic'] = data['Comment'].apply(
        lambda x: comment_to_topic(x, topics)
    )

    return data
    
def comment_to_topic(comment, topics):
    candidate_topics = []
    for topic, comments in topics.items():
        if comment in comments:
            candidate_topics.append(topic)
    
    if candidate_topics:
        # Create a TF-IDF vectorizer and fit it on the comment and candidate topics
        vectorizer = TfidfVectorizer().fit_transform([comment] + candidate_topics)
        vectors = vectorizer.toarray()
        
        # Calculate cosine similarities between the comment vector and each candidate topic vector
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Find the topic with the highest cosine similarity
        best_topic_index = cosine_similarities.argmax()

        return candidate_topics[best_topic_index]
    else:
        return 'No topic'

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:

    if file == 'excluded_states.csv':
        continue

    data = pd.read_csv(f'{folder}/{file}')
    state = file.split('.')[0]
    data = state_comments_to_topics(data, state)
    data.to_csv(f'{folder}/{file}', index=False)
