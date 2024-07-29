from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import ParameterGrid
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
        top_words = [vocab[id] for id in comp.argsort()][::-1][:10]
        topic_words.append(top_words)
        topics_list[f"Topic {i + 1}"] = top_words

    if len(topics_list) == 0:
        return None, None, topics_list

    return calculate_coherence_score(topic_words), calculate_average_jaccard_similarity(topic_words), topics_list

def find_best_topics(comments):

    param_grid = {
        'n_components': [5, 10, 15],
        'max_features': [200, 500, 1000],
        'ngram_range': [(1, 3), (2, 4), (3, 5)]
    }

    best_score = -1
    best_coherence = 0
    best_jaccard = 0
    best_topics = None
    best_params = None

    for params in ParameterGrid(param_grid):
        coherence, jaccard, topics = topic_modeling(
            comments, 
            n_components=params['n_components'],
            max_features=params['max_features'], 
            ngram_range=params['ngram_range']
        )
        if coherence and jaccard:
            score = coherence - jaccard
            if score > best_score:
                best_score = score
                best_coherence = coherence
                best_jaccard = jaccard
                best_topics = topics
                best_params = params

    return best_coherence, best_jaccard, best_params, best_topics

if __name__ == '__main__':

    # Comments by state
    folder1 = 'csv_files/classified_comments_by_state'
    files1 = os.listdir(folder1)
    
    # negative comments by state
    folder2 = 'csv_files/negative_comments_by_state'
    files2 = os.listdir(folder2)

    files = []
    for file1, file2 in zip(files1, files2):
        files.append((file1, file2))

    coherence_jaccard1 = []
    coherence_jaccard2 = []

    # Iterate through the files and find the best topics for each state 
    for file1, file2 in files:
        
        state = file1[:-4]
        state_data = pd.read_csv(f'{folder1}/{file1}')
        state_comment = state_data['Comment'].tolist()

        best_coherence, best_jaccard, best_params, best_topics = find_best_topics(state_comment)
        coherence_jaccard1.append(f'{state}: Coherence - {best_coherence}, Jaccard - {best_jaccard}, Params - {best_params}')

        # Write the results to a file
        with open(f'topic_modeling/LDA_method/topics_by_state/{state}.json', 'w') as file:
            json.dump(best_topics, file)

        state = file2[:-4]
        state_data = pd.read_csv(f'{folder2}/{file2}')
        state_comment = state_data['Comment'].tolist()

        best_coherence, best_jaccard, best_params, best_topics = find_best_topics(state_comment)
        coherence_jaccard2.append(f'{state}: Coherence - {best_coherence}, Jaccard - {best_jaccard}, Params - {best_params}')

        # Write the results to a file
        with open(f'topic_modeling/LDA_method/negative_topics_by_state/{state}.json', 'w') as file:
            json.dump(best_topics, file)

    # Save the coherence and jaccard scores to files
    with open(f'topic_modeling/LDA_method/detailed_info_overall.txt', 'w') as file:
        file.write('\n'.join(coherence_jaccard1))

    with open(f'topic_modeling/LDA_method/detailed_info_negative_comments.txt', 'w') as file:
        file.write('\n'.join(coherence_jaccard2))