from bertopic import BERTopic

topic_model = BERTopic.load("reddit_data/topic_modeling/topic_model")

fig1 = topic_model.visualize_topics()
fig2 = topic_model.visualize_barchart(top_n_topics=len(topic_model.get_topics()))

fig1.write_image("graphs/intertopic_distances.png")
fig2.write_image("graphs/topics_barchart.png")