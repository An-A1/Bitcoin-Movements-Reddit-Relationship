import praw
from tqdm import tqdm
from praw.models import Submission
from datetime import datetime, timezone
import pandas as pd
import os

def get_reddit_data(subreddits, limit, csv_file_path):
    REDDIT_ID = 'pVf7UANnO4BViQz8dGZtiQ'
    REDDIT_SECRET = 'I40zsP4SesKT2xW3MFVWuPx6lQc4EA'
    USER_AGENT = "MyApp/1.0 by u/bolis_hakim"

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
                
                results.append({**data, **comment_data})

        results_df = pd.DataFrame(results)
        all_results = pd.concat([all_results, results_df], ignore_index=True)
        all_results.drop_duplicates(subset=['comment_id','post_id'], inplace=True)
        all_results = all_results[all_results['upvotes'] >= 10]
        all_results.to_csv(csv_file_path, index=False, encoding='utf-8')

