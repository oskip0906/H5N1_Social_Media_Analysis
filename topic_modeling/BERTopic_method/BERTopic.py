from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid
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

def topic_modeling(comments, min_cluster_size, min_samples, max_features, ngram_range):

    processed_comments = []
    for comment in comments:
        processed_comments.append(preprocess_comment(comment))

    # HDBSCAN model for clustering 
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        cluster_selection_epsilon=0.1
    )

    # BERTopic model with adjusted parameters
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model, 
        embedding_model='all-MiniLM-L6-v2', 
        vectorizer_model=CountVectorizer(max_features=max_features, ngram_range=ngram_range),
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
        return None, topics_list

    return calculate_coherence_score(list(topics_list.values())), topics_list

if __name__ == '__main__':

    # Data by state
    folder = 'csv_files/classified_comments_by_state'
    files = os.listdir(folder)

   # Define the parameter grid
    param_grid = {
        'min_cluster_size': [2, 4],
        'min_samples': [2, 4],
        'max_features': [500, 1000, 1500],
        'ngram_range': [(1, 3), (2, 4), (3, 5)]
    }

    coherence_scores = []

    # Iterate through the files and find the best topics for each state 
    for file in files:
        state = file[:-4]
        state_data = pd.read_csv(f'{folder}/{file}')
        state_comment = state_data['Comment'].tolist()

        best_score = -1
        best_params = None
        best_topics = None

        for params in ParameterGrid(param_grid):
            score, topics = topic_modeling(
                state_comment, 
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                max_features=params['max_features'],
                ngram_range=params['ngram_range']
            )
            if score is not None and score > best_score:
                best_score = score
                best_params = params
                best_topics = topics

        coherence_scores.append(f'{state}: {best_score} with params {best_params}')

        # Save the best topics to a JSON file for each state
        with open(f'topic_modeling/BERTopic_method/topics_by_state/{state}.json', 'w') as f:
            json.dump(best_topics, f, indent=4)

    # Save the coherence scores to a file
    with open(f'topic_modeling/BERTopic_method/coherence_scores.txt', 'w') as file:
        file.write('\n'.join(coherence_scores))