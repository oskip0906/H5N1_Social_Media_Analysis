from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ParameterGrid
import hdbscan 
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import itertools
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

def calculate_jaccard_similarity(topic1, topic2):

    set1, set2 = set(topic1), set(topic2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if union:
        return len(intersection) / len(union)
    
    return 0

def calculate_average_jaccard_similarity(topics):
    
    total_similarity = 0
    num_comparisons = 0
    
    # Compare every pair of topics
    for topic1, topic2 in itertools.combinations(topics, 2):
        total_similarity += calculate_jaccard_similarity(topic1, topic2)
        num_comparisons += 1

    if num_comparisons > 0:
        return total_similarity / num_comparisons
    
    return 0

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
    topic_words = []

    i = 0
    while True:
        topic = topic_model.get_topic(i)
        if not topic:
            break
        subtopics_list = [word for word, _ in topic]
        topics_list[f"Topic {i + 1}"] = subtopics_list
        topic_words.append(subtopics_list)
        i += 1

    if len(topics_list) == 0:
        return None, None, topics_list

    return calculate_coherence_score(topic_words), calculate_average_jaccard_similarity(topic_words), topics_list

if __name__ == '__main__':

    # Data by state
    folder = 'csv_files/classified_comments_by_state'
    files = os.listdir(folder)

    # Define the parameter grid
    param_grid = {
        'min_cluster_size': [2, 5],
        'min_samples': [2, 5],
        'max_features': [200, 500, 1000],
        'ngram_range': [(1, 3), (2, 4), (3, 5)]
    }

    coherence_jaccard = []

    # Iterate through the files and find the best topics for each state 
    for file in files:
        state = file[:-4]
        state_data = pd.read_csv(f'{folder}/{file}')
        state_comment = state_data['Comment'].tolist()

        best_score = -1
        best_params = None
        best_topics = None
        best_coherence = None
        best_jaccard = None

        for params in ParameterGrid(param_grid):
            coherence, jaccard, topics = topic_modeling(
                state_comment, 
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                max_features=params['max_features'],
                ngram_range=params['ngram_range']
            )
            if coherence is not None and jaccard is not None:
                score = coherence - jaccard
                if score > best_score:
                    best_score = score
                    best_coherence = coherence
                    best_jaccard = jaccard
                    best_params = params
                    best_topics = topics

        coherence_jaccard.append(f'{state}: {best_coherence}, {best_jaccard}, with params {best_params}')

        # Save the best topics to a JSON file for each state
        with open(f'topic_modeling/BERTopic_method/topics_by_state/{state}.json', 'w') as f:
            json.dump(best_topics, f, indent=4)

    # Save the coherence and jaccard scores to a file
    with open(f'topic_modeling/BERTopic_method/detailed_info.txt', 'w') as file:
        file.write('\n'.join(coherence_jaccard))