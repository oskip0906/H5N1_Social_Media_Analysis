from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

topic_model = BERTopic.load("models/topic_model_1")

# extract topic representations
topics = topic_model.get_topics()
topic_representations = []

for topic_id in range(len(topics)):
    words = topic_model.get_topic(topic_id)
    if words:
        topic_representations.append(' '.join([word[0] for word in words]))

# vectorize the topic representations
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(topic_representations)

# calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# calculate the average pairwise cosine similarity; ignoring self-similarity
n = similarity_matrix.shape[0]
sum_similarities = similarity_matrix.sum() - n  
avg_similarity = sum_similarities / (n * (n - 1)) 

# Print the coherence score
print(f"Topic Coherence Score: {avg_similarity}")