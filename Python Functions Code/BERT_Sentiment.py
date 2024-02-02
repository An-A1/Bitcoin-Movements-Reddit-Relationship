from transformers import pipeline
import pandas as pd
from tqdm import tqdm

def BERT_sentiments(input_csv, output_csv, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", max_token_length=512):
    sentiment_pipeline = pipeline(model=model_name)
    
    # Read the input CSV file
    Reddit_data = pd.read_csv(input_csv)

    # Initialize new columns
    Reddit_data['BERT_class'] = "Neutral"
    Reddit_data['BERT_Compound'] = 0.0

    for i in tqdm(range(len(Reddit_data['comment_body'])), desc="Processing Sentiments"):
        comment = Reddit_data['comment_body'][i][:max_token_length] 
        sentiment_dict = sentiment_pipeline(comment)
        Reddit_data['BERT_class'][i] = sentiment_dict[0]['label'].upper()
        Reddit_data['BERT_Compound'][i] = sentiment_dict[0]['score']
        

    # Remove rows with 'BERT_Compound' equal to 0.0
    Reddit_data = Reddit_data[Reddit_data['BERT_Compound'] != 0.0]
    Reddit_data.to_csv(output_csv, index=False)
    data = pd.read_csv(output_csv)

    # Calculate percentage for each sentiment class
    sentiment_percentage = data["BERT_class"].value_counts(normalize=True) * 100
    print("Sentiment Percentage:")
    print(sentiment_percentage)

    # Calculate mean for each sentiment class
    mean_sentiments = data.groupby('BERT_class')['BERT_Compound'].mean()
    print("Mean Sentiments:")
    print(mean_sentiments)



# Example usage:
# BERT_sentiments(input_csv='Reddit_new.csv', output_csv='BERT_sentiment_data.csv')
