import praw
from tqdm import tqdm
from praw.models import Submission
from datetime import datetime, timezone
import pandas as pd
import os

def get_reddit_data(subreddits, limit, csv_file_path, filter_words):
    REDDIT_ID = ''
    REDDIT_SECRET = ''
    USER_AGENT = ''

    # Reddit API credentials
    reddit = praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=USER_AGENT
    )

    # Check if the CSV file exists
    if os.path.isfile(csv_file_path):
        all_results = pd.read_csv(csv_file_path)

    else:
        all_results = pd.DataFrame()

    for sub in tqdm(subreddits):
        results = []
        selected_threads = [
            submission for submission in reddit.subreddit(sub).hot(limit=limit)
            if not (submission.pinned or submission.stickied)
            and isinstance(submission, Submission)
        ]

        for post in selected_threads:
            post_date = datetime.fromtimestamp(post.created_utc, timezone.utc)
            data = {
                'author_name': post.author.name if post.author else 'No author',
                'post_id': post.id,
                'title': post.title,
                'body': post.selftext,
                'post_date': post_date,
                'upvotes': post.ups,
            }

            # Get comments for the post
            post.comments.replace_more(limit=None)
            comments = post.comments.list()

            for comment in comments:
                comment_date = datetime.fromtimestamp(comment.created_utc, timezone.utc)
                comment_data = {
                    'comment_id': comment.id,
                    'comment_parent_id': comment.parent_id,
                    'comment_body': comment.body,
                    'comment_date': comment_date,
                    'subreddit': sub,
                }
                
                # Check if the comment contains any of the filter words
                if any(word in comment.body.lower() for word in filter_words):
                    results.append({**data, **comment_data})

        results_df = pd.DataFrame(results)
        all_results = pd.concat([all_results, results_df], ignore_index=True)
        all_results.drop_duplicates(subset=['comment_id','post_id'], inplace=True)
        #all_results = all_results[all_results['upvotes'] >= 10]
        all_results.to_csv(csv_file_path, index=False, encoding='utf-8')

# Example usage
subreddit_list = ["Bitcoin", "BitcoinMarkets"]
csv_path = 'Reddit_new.csv'
filter_words_list = [
    'bitcoin', 'btc', 'crypto', 'cryptocurrency',
    'price', 'value', 'market', 'exchange',
    'buy', 'sell', 'trade', 'trading',
    'invest', 'investment', 'investor',
    'rise', 'increase', 'up', 'bullish',
    'fall', 'decrease', 'down', 'bearish',
    'prediction', 'forecast', 'analyze',
    'chart', 'technical analysis', 'TA',
    'ATH', 'all-time high', 'dip', 'pump',
    'dump', 'volatility', 'swing', 'speculation',
    'HODL', 'FOMO', 'FUD', 'moon', 'lambo',
    'wallet', 'blockchain', 'hashrate',
    'mining', 'halving', 'hard fork', 'soft fork',
    'crash', 'expect', 'money', 'dollar',
    'period', 'guess', 'see', 'point',
    'duration', 'time', 'exit', 'out',
    'short', 'hold','sentiment', 'feel', 'emotion',
    'positive', 'negative', 'optimistic', 'pessimistic', 
    'hope', 'fear'
]

# get_reddit_data(subreddit_list, limit=30, csv_file_path=csv_path, filter_words=filter_words_list)
