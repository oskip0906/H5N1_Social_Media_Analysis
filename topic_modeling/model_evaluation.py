from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

def calculate_coherence_score(model):

    # extract topic representations
    topics = model.get_topics()
    topic_representations = []

    for topic_id in range(len(topics)):
        words = model.get_topic(topic_id)
        if words:
            topic_representations.append(' '.join([word[0] for word in words]))

    # use SentenceTransformer for vectorization to capture semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(topic_representations)

    # calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # calculate the average pairwise cosine similarity; ignoring self-similarity
    n = similarity_matrix.shape[0]
    sum_similarities = similarity_matrix.sum() - n  
    avg_similarity = sum_similarities / (n * (n - 1)) 
    
    return avg_similarity

# load the topic models
folder = 'reddit_data/topic_modeling/topic_models'
files = os.listdir(folder)

coherence_scores = []

for file in files:
    state = file.split('/')[-1].split('_')[0]
    model = BERTopic.load(f'{folder}/{file}')
    coherence_scores.append(f'{state}: {calculate_coherence_score(model)}')

with open('reddit_data/topic_modeling/coherence_scores.txt', 'w') as file:
    file.write('\n'.join(coherence_scores))