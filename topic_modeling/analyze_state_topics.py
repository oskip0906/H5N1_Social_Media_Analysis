import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from collections import Counter

def load_topic_data():
    # Path to the topic files
    bertopic_path = 'topic_modeling/BERTopic_method/topics_by_state'
    
    # Dictionary to store all topics
    all_topics = {}
    
    # Load all state topic files
    for file in os.listdir(bertopic_path):
        if file.endswith('.json'):
            state = file.split('.')[0]
            with open(os.path.join(bertopic_path, file), 'r') as f:
                topics = json.load(f)
                all_topics[state] = topics
    
    return all_topics

def identify_top_topic_per_state(all_topics):
    top_topics_by_state = {}
    
    for state, topics in all_topics.items():
        if not topics:  # Skip empty topic dictionaries
            continue
            
        # Count frequency of each topic in the state
        topic_counter = Counter(topics.keys())
        
        # Get the top topic for this state
        top_topic = topic_counter.most_common(1)
        if top_topic:  # Check if we have any topics
            top_topics_by_state[state] = top_topic[0][0]
    
    return top_topics_by_state

def load_classified_comments():
    comments_path = 'csv_files/classified_comments.csv'
    return pd.read_csv(comments_path)

def load_topic_assignments():
    # Path to the classified comments by state
    state_path = 'csv_files/classified_comments_by_state'
    
    # Load and combine all state files
    all_comments = []
    for file in os.listdir(state_path):
        # Skip the excluded_states.csv file
        if file == 'excluded_states.csv':
            continue
            
        if file.endswith('.csv'):
            file_path = os.path.join(state_path, file)
            try:
                state_comments = pd.read_csv(file_path)
                if 'corresponding_topic' in state_comments.columns:
                    # Add state as a column to each dataframe
                    state_comments['state'] = file.replace('.csv', '')
                    all_comments.append(state_comments)
                else:
                    print(f"Warning: 'corresponding_topic' column not found in {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    # Combine all comments
    if all_comments:
        comments = pd.concat(all_comments, ignore_index=True)
        print(f"Loaded {len(comments)} comments with topic assignments from state-specific files")
        return comments
    else:
        # Fallback if no state files have the column
        print("Warning: No state files with 'corresponding_topic' column found.")
        return None

def analyze_top_topic_per_state(top_topics_by_state):
    # Load comments with topic assignments
    comments = load_topic_assignments()
    if comments is None:
        print("Error: Could not load comments with topic assignments.")
        return

    num_states = len(top_topics_by_state)
    num_cols = 3
    num_rows = (num_states + num_cols - 1) // num_cols  # Ceiling division to get enough rows

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, num_rows * 8), constrained_layout=True)
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]
    
    # Common sentiment categories - exclude Neutral, Surprise, Love
    sentiment_categories = ['Sadness', 'Fear', 'Anger', 'Joy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Analyze each state's top topic
    for i, (state, topic) in enumerate(top_topics_by_state.items()):
        # The format in the CSV is "BERTopic Topic X" 
        topic_number = topic.replace("Topic ", "").strip()
        bertopic_name = f"BERTopic Topic {topic_number}"
        
        # Filter comments for this state and top topic
        state_comments = comments[comments['state'] == state]
        topic_comments = state_comments[state_comments['corresponding_topic'] == bertopic_name]
        
        # Count sentiments for this topic, filtering for only our selected sentiments
        sentiment_counts = topic_comments[topic_comments['Sentiment'].isin(sentiment_categories)]['Sentiment'].value_counts()
        
        # Make sure all categories are present
        for cat in sentiment_categories:
            if cat not in sentiment_counts:
                sentiment_counts[cat] = 0
        
        # Sort by categories for consistency
        sentiment_counts = sentiment_counts.reindex(sentiment_categories).fillna(0)
        
        # Check if we have any non-zero data to plot
        total = sentiment_counts.sum()
        
        # Create pie chart
        ax = axes[i]
        
        if total > 0:
            # We have data to plot
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            # Customize text properties
            for text in texts:
                text.set_size(20)
            for autotext in autotexts:
                autotext.set_size(20)
                autotext.set_color('white')
            
            # Set title with state, topic and total count
            ax.set_title(f"{state.capitalize()} (Total: {total})", fontsize=20)
        else:
            # No data available for this topic
            ax.text(0.5, 0.5, "No sentiment data available", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f"{state.capitalize()} (Total: 0)", fontsize=20)
            ax.axis('off')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Save the figure
    os.makedirs('graphs/topic_sentiment_analysis', exist_ok=True)
    plt.savefig('graphs/topic_sentiment_analysis/top_topic_per_state_sentiment_distribution.png', dpi=300)
    plt.close()
    
    print(f"Created pie charts for the top topic of each state.")

def export_top_topic_sentiment_data(top_topics_by_state):
    sentiment_categories = ['Sadness', 'Fear', 'Anger', 'Joy']
    
    # Load comments with topic assignments
    comments = load_topic_assignments()
    if comments is None:
        print("Error: Could not load comments with topic assignments.")
        return
    
    # Prepare data
    results = []
    
    for state, topic in top_topics_by_state.items():
        # The format in the CSV is "BERTopic Topic X" 
        topic_number = topic.replace("Topic ", "").strip()
        bertopic_name = f"BERTopic Topic {topic_number}"
        
        # Filter comments for this state and top topic
        state_comments = comments[comments['state'] == state]
        topic_comments = state_comments[state_comments['corresponding_topic'] == bertopic_name]
        
        # Count sentiments for this topic, filtering for only our selected sentiments
        sentiment_counts = topic_comments[topic_comments['Sentiment'].isin(sentiment_categories)]['Sentiment'].value_counts()
        
        # Make sure all categories are present
        for cat in sentiment_categories:
            if cat not in sentiment_counts:
                sentiment_counts[cat] = 0
        
        # Sort by categories for consistency
        sentiment_counts = sentiment_counts.reindex(sentiment_categories).fillna(0)
        
        # Create a row for each sentiment
        for sentiment, count in sentiment_counts.items():
            results.append({
                'State': state,
                'Topic': topic,
                'Sentiment': sentiment,
                'Count': count
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create directory if it doesn't exist
    os.makedirs('csv_files/topic_sentiment_analysis', exist_ok=True)
    
    # Save to CSV
    results_df.to_csv('csv_files/topic_sentiment_analysis/top_topic_per_state_sentiment_distribution.csv', index=False)
    
    print(f"Exported sentiment distribution data to csv_files/topic_sentiment_analysis/top_topic_per_state_sentiment_distribution.csv")

def main():
    print("Loading topic data...")
    all_topics = load_topic_data()
    
    print("Identifying top topic for each state...")
    top_topics_by_state = identify_top_topic_per_state(all_topics)
    print(f"Top topics by state: {top_topics_by_state}")
    
    print("Loading classified comments...")
    comments = load_topic_assignments()
    
    print("Analyzing sentiment distribution for top topic of each state...")
    analyze_top_topic_per_state(top_topics_by_state)
    
    print("Exporting sentiment data...")
    export_top_topic_sentiment_data(top_topics_by_state)
    
    print("Analysis complete. Results saved to graphs/topic_sentiment_analysis/")

if __name__ == "__main__":
    main()
