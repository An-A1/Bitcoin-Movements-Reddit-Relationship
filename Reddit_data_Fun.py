import praw
import csv
from tqdm import tqdm
from praw.models import Submission
from datetime import datetime, timezone

def get_reddit_data(subreddits, limit, csv_file_path):
    REDDIT_ID ="Your Client ID"
    REDDIT_SECRET = "Your Client Secret"
    USER_AGENT = "User Agent"

    # Reddit API credentials
    reddit = praw.Reddit(
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=USER_AGENT
    )

    all_results = []

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
            }

            # Get comments for the post
            post.comments.replace_more(limit=None)
            comments = post.comments.list()

            for comment in comments:
                comment_data = {
                    'comment_id': comment.id,
                    'comment_parent_id': comment.parent_id,
                    'comment_body': comment.body,
                    'subreddit': sub,
                }
                results.append({**data, **comment_data})

        all_results.extend(results)

    # Write data to a single CSV file
    with open(csv_file_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['author_name', 'post_id', 'title', 'body', 'post_date', 'comment_id', 'comment_parent_id', 'comment_body', 'subreddit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
