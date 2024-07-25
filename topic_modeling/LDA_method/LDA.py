from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import ParameterGrid
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

def topic_modeling(comments, n_components, max_features, ngram_range):

    processed_comments = []
    for comment in comments:
        processed_comments.append(preprocess_comment(comment))

    vect = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    vect_text = vect.fit_transform(processed_comments)

    # Initialize LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_components,
        learning_method='online', 
        random_state=42, 
        max_iter=10
    )
    
    lda_model.fit_transform(vect_text)
    vocab = vect.get_feature_names_out()

    topic_words = []
    topics_list = {}

    # Get the top 10 words for each topic
    for i, comp in enumerate(lda_model.components_):
        top_words = [vocab[id] for id in comp.argsort()]
        topic_words.append(top_words)
        topics_list[f"Topic {i + 1}"] = top_words

    if len(topics_list) == 0:
        return None, topics_list

    return calculate_coherence_score(list(topics_list.values())), topics_list

if __name__ == '__main__':

    # Data by state
    folder = 'csv_files/classified_comments_by_state'
    files = os.listdir(folder)

    # Define the parameter grid
    param_grid = {
        'n_components': [5, 10, 15],
        'max_features': [500, 1000, 1500],
        'ngram_range': [(1, 3), (2, 4), (3, 5)]
    }

    coherence_scores = []

    # Iterate through the files and find the best topics for each state 
    for file in files:
        state = file[:-4].split('/')[-1]
        state_data = pd.read_csv(f'{folder}/{file}')
        state_comment = state_data['Comment'].tolist()
        
        best_score = -1
        best_params = None
        best_topics = None

        for params in ParameterGrid(param_grid):
            score, topics = topic_modeling(
                state_comment, 
                n_components=params['n_components'],
                max_features=params['max_features'],
                ngram_range=params['ngram_range']
            )
            if score is not None and score > best_score:
                best_score = score
                best_params = params
                best_topics = topics

        coherence_scores.append(f'{state}: {best_score} with params {best_params}')

        # Save the best topics to a JSON file for each state
        with open(f'topic_modeling/LDA_method/topics_by_state/{state}.json', 'w') as f:
            json.dump(best_topics, f, indent=4)

    # Save the coherence scores to a file
    with open(f'topic_modeling/LDA_method/coherence_scores.txt', 'w') as file:
        file.write('\n'.join(coherence_scores))