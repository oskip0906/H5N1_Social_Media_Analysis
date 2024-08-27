import os
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def state_comments_to_topics(data, state):

    with open(f'topic_modeling/BERTopic_method/topics_by_state/{state}.json', 'r') as f:
        topics1 = json.load(f)

    with open(f'topic_modeling/LDA_method/topics_by_state/{state}.json', 'r') as f:
        topics2 = json.load(f)

    last_key_for_topics1 = list(topics1.keys())[-1]
    last_topic_number = int(last_key_for_topics1.split(' ')[1])

    topics2 = {f'Topic {last_topic_number + i}': topics2[key] for i, key in enumerate(topics2)}

    topics = {**topics1, **topics2}

    with open(f'topic_modeling/combined_topics_by_state/{state}.json', 'w') as f:
        json.dump(topics, f)

    data[['corresponding_topic', 'similarity_to_topic']] = data['Comment'].apply(
        lambda x: pd.Series(comment_to_topic(x, topics))
    )

    return data

def comment_to_topic(comment, topics):
  
    topic_texts = []

    for subtopics in topics.values():
        filtered_subtopic = ""
        for word in " ".join(subtopics).split():
            if word.lower() not in ['bird', 'avian', 'flu', 'influenza', 'h5n1']:
                filtered_subtopic += word + " "
        topic_texts.append(filtered_subtopic.strip())

    # Vectorize text data
    vectorizer = CountVectorizer()
    matrix1 = vectorizer.fit_transform([comment])
    matrix2 = vectorizer.transform(topic_texts)

    # Calculate cosine similarity between comment and topic
    similarities = cosine_similarity(matrix1, matrix2).flatten()
    
    # Find topic with highest similarity
    highest_similarity_index = similarities.argmax()
    assigned_topic = list(topics.keys())[highest_similarity_index]
    
    return assigned_topic, similarities[highest_similarity_index]

folder = 'csv_files/classified_comments_by_state'
files = os.listdir(folder)

for file in files:
    data = pd.read_csv(f'{folder}/{file}')
    state = file.split('.')[0]
    data = state_comments_to_topics(data, state)
    data.to_csv(f'{folder}/{file}', index=False)