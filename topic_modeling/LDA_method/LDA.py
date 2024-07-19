from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pickle
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

    vect = CountVectorizer(max_features=1000, ngram_range=(2, 5))
    vect_text = vect.fit_transform(processed_comments)

    # Initialize LDA model
    lda_model = LatentDirichletAllocation(
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
        # Top 10 words for each topic
        top_words = [vocab[id] for id in comp.argsort()[-10:]]
        topic_words.append(top_words)
        topics_list[f"Topic {i + 1}"] = top_words

    # save the topics to a JSON file
    with open(f'topic_modeling/LDA_method/topics_by_state/{output}.json', 'w') as f:
        json.dump(topics_list, f, indent=4)

    # save the models
    with open(f'topic_modeling/LDA_method/models_by_state/{output}_topic_model', 'wb') as model_file:
        pickle.dump(lda_model, model_file)

    # Calculate and add the coherence score
    coherence_score = calculate_coherence_score(topic_words)
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
    with open(f'topic_modeling/LDA_method/coherence_scores.txt', 'w') as file:
        file.write('\n'.join(coherence_scores))