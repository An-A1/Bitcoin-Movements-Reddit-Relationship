from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tqdm import tqdm

def VADER_sentiment_analysis(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    data = data.dropna(subset=['comment_body'])
    data['comment_body'] = data['comment_body'].astype(str)
    vader = SentimentIntensityAnalyzer()

    data["VADER_compound"] = 0.0
    for i in tqdm(range(len(data['comment_body'])), desc="Analyzing Sentiments"):
        # use polarity_scores method to get the sentiment scores
        sentiment_dict = vader.polarity_scores(data['comment_body'][i])
        # Save the compound result as float in the dataset.
        data["VADER_compound"][i] = float(sentiment_dict['compound'])

    data["VADER_class"] = ""
    for i in tqdm(range(len(data['VADER_compound'])), desc="Assigning Classes"):
        compound_score = data['VADER_compound'][i]
        if compound_score > 0.5:
            data["VADER_class"][i] = "Positive"
        elif compound_score < -0.5:
            data["VADER_class"][i] = "Negative"
        else:
            data["VADER_class"][i] = "Neutral"

    # Calculate percentage for each sentiment class
    sentiment_percentage = data["VADER_class"].value_counts(normalize=True) * 100
    print("Sentiment Percentage:")
    print(sentiment_percentage)

    # Calculate mean for each sentiment class
    mean_sentiments = data.groupby('VADER_class')['VADER_compound'].mean()
    print("Mean Sentiments:")
    print(mean_sentiments)

    data = data[data['VADER_compound'] != 0]
    data.to_csv(output_csv, index=False)

# Example usage:
# VADER_sentiment_analysis(input_csv='Reddit_new.csv', output_csv='VADER_sentiment_data.csv')
