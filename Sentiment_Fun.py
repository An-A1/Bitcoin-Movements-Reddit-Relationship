from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = text.replace('http://', '').replace('https://', '').replace('www.', '')
    # Remove non-English characters (a basic check)
    if not all(ord(char) < 128 for char in text):
        text = ''
    # Remove special characters
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)

    return text.strip()



def perform_sentiment_analysis(file_path):
    data = pd.read_csv(file_path)
    data['comment_body'] = data['comment_body'].apply(clean_text)
    data = data.dropna(subset=['comment_body'])
    data['comment_body'] = data['comment_body'].astype(str)
    vader = SentimentIntensityAnalyzer()

    data["VADER_compound"] = 0.0
    for i in range(len(data['comment_body'])):
        # use polarity_scores method to get the sentiment scores
        sentiment_dict = (vader.polarity_scores(data['comment_body'][i]))
        # Save the compound result as float in the dataset.
        data["VADER_compound"][i] = float(sentiment_dict['compound'])

    data["VADER_class"] = ""
    for i in range(len(data['VADER_compound'])):
        compound_score = data['VADER_compound'][i]
        if compound_score > 0.5:
            data.loc[i, "VADER_class"] = "Positive"
        elif compound_score < -0.5:
            data.loc[i, "VADER_class"] = "Negative"
        else:
            data.loc[i, "VADER_class"] = "Neutral"

    # Calculate percentage for each sentiment class
    sentiment_percentage = data["VADER_class"].value_counts(normalize=True) * 100
    print("Sentiment Percentage:")
    print(sentiment_percentage)

    # Calculate mean for each sentiment class
    mean_sentiments = data.groupby('VADER_class')['VADER_compound'].mean()
    print("Mean Sentiments:")
    print(mean_sentiments)

    data.to_csv(f"{file_path}", index=False)
