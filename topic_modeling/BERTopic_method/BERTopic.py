from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan 
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pandas as pd
import json
import os

import sys
sys.path.append('topic_modeling')
from process_comments import preprocess_comment

def calculate_coherence_score(topic_words):

    # Create a dictionary
    dictionary = Dictionary(topic_words)
    
    # Initialize coherence model
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=topic_words, 
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    return coherence_score

def topic_modeling(coherence_scores, comments, output):

    processed_comments = []
    for comment in comments:
        processed_comments.append(preprocess_comment(comment))

    # HDBSCAN model for clustering 
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5, 
        min_samples=5, 
        cluster_selection_epsilon=0.5
    )

    # BERTopic model with adjusted parameters
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model, 
        embedding_model='all-MiniLM-L6-v2', 
        vectorizer_model=CountVectorizer(max_features=1000, ngram_range=(2, 5)),
        low_memory=True
    )

    # fit-transform the model
    topic_model.fit_transform(processed_comments)
    
    topics_list = {}

    i = 0
    while topic_model.get_topic(i) != False:
        subtopics = topic_model.get_topic(i)
        subtopics_list = []
        for subtopic in subtopics:
            subtopics_list.append(subtopic[0])
        topics_list[f"Topic {i + 1}"] = subtopics_list
        i += 1

    if len(topics_list) == 0:
        return

    # save the topics to a JSON file
    with open(f'topic_modeling/BERTopic_method/topics_by_state/{output}.json', 'w') as f:
        json.dump(topics_list, f, indent=4)

    # save the model
    topic_model.save(f'topic_modeling/BERTopic_method/models_by_state/{output}_topic_model')

    # Calculate and add the coherence score
    coherence_score = calculate_coherence_score(topics_list.values())
    coherence_scores.append(f'{output}: {coherence_score}')

if __name__ == '__main__':

    # data by state
    folder = 'csv_files/classified_comments_by_state'
    files = os.listdir(folder)

    coherence_scores = []

    for file in files:
        state = file[:-4].split('/')[-1]
        state_data = pd.read_csv(f'{folder}/{file}')
        state_comment = state_data['Comment'].tolist()
        topic_modeling(coherence_scores, state_comment, state)

    # Save the coherence scores to a file
    with open(f'topic_modeling/BERTopic_method/coherence_scores.txt', 'w') as file:
        file.write('\n'.join(coherence_scores))