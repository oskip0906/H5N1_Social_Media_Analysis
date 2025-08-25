import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def count_comments_by_subreddit():
    """
    Counts the number of comments for each subreddit in the dataset.
    Returns a DataFrame sorted by comment count in descending order.
    """
    # Load the comments data
    comments_path = 'csv_files/comments.csv'
    df = pd.read_csv(comments_path)
    
    # Count comments by subreddit
    subreddit_counts = df['Subreddit'].value_counts().reset_index()
    subreddit_counts.columns = ['Subreddit', 'Comment_Count']
    
    return subreddit_counts

def create_subreddit_bar_plot(subreddit_counts, top_n=10):
    """
    Creates a bar plot showing the count of comments for the top N subreddits.
    
    Args:
        subreddit_counts: DataFrame with subreddit counts
        top_n: Number of top subreddits to include in the plot (default: 15)
    """
    top_subreddits = subreddit_counts.head(top_n)
    
    plt.figure(figsize=(14, 7))
    
    # Create a bar plot
    bars = plt.bar(top_subreddits['Subreddit'], top_subreddits['Comment_Count'])
    
    # Add labels and title
    plt.xlabel('Subreddit', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    plt.title(f'Top {top_n} Subreddits by Comment Count', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:,}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout to make room for labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('graphs/subreddit_comment_counts.png', dpi=300)
    plt.close()
    
    print(f"Bar plot saved as 'graphs/subreddit_comment_counts.png'")


if __name__ == "__main__":
    subreddit_counts = count_comments_by_subreddit()
    print(f"Found {len(subreddit_counts)} subreddits with comments")
    print("\nTop 10 subreddits by comment count:")
    print(subreddit_counts.head(10).to_string(index=False))
    
    # Create and save the bar plot
    create_subreddit_bar_plot(subreddit_counts)
