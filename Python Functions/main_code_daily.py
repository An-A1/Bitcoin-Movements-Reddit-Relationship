import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from Reddit_data_Fun import *
from Cryptocurrency_Fun import *
from VADER_Sentiment import *



# First to Run the collection of Rediit Data
get_reddit_data(["Bitcoin", "BitcoinMarkets"], 20, "all_subreddits_data.csv")

#Call the function to Perform the sentiment analysis in the Redditthe DataFrame
perform_sentiment_analysis('all_subreddits_data.csv')

#Load Reddit Data for perform the merge and Corrolation
Reddit_data = pd.read_csv('VADER_all_subreddits_data.csv.csv')

# Group and aggregate data
Reddit_data['comment_date'] = pd.to_datetime(Reddit_data['comment_date'])
Reddit_data['comment_date'] = Reddit_data['comment_date'].dt.strftime('%Y-%m-%d')
Reddit_data = Reddit_data.groupby('comment_date').agg(VADER_mean=('VADER_compound', 'mean')).reset_index()

# Second to Run the collection of Cryptocurrency price action
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y.%m.%d')
end_date = (datetime.now()).strftime('%Y.%m.%d')
symbol = 'BTC/USDT'
timeframe = '1d'
crypto_data = fetch_crypto_data(symbol, start_date, end_date, timeframe)


# Load Cryptocurrency price data
price_data = pd.read_csv(f'{symbol.replace("/", "_")}_data.csv')


# Merge data on the common column ('comment_date' and 'timestamp')
merged_data = pd.merge(Reddit_data, price_data, left_on='comment_date', right_on='date', how='inner')


# Save the DataFrame to a CSV file
csv_filename = f'merged_data.csv'
merged_data.to_csv(csv_filename, index=False)

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation = merged_data['VADER_mean'].corr(merged_data['volume'])
print(f"Correlation between sentiment and Bitcoin Trading Volume: {correlation}")

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation = merged_data['VADER_mean'].corr(merged_data['close'])
print(f"Correlation between sentiment and Bitcoin price: {correlation}")

# Plot the correlation
plt.scatter(merged_data['VADER_mean'], merged_data['volume'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['VADER_mean'], merged_data['volume'], 1)
plt.plot(merged_data['VADER_mean'], m*merged_data['VADER_mean']+b, color='red', label='Regression Line')
plt.title('Correlation between Sentiment and Trading Volume')
plt.xlabel('Sentiment (VADER_mean)')
plt.ylabel('Bitcoin Price (close)')
plt.show()

# Plot the correlation
plt.scatter(merged_data['VADER_mean'], merged_data['close'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['VADER_mean'], merged_data['close'], 1)
plt.plot(merged_data['VADER_mean'], m*merged_data['VADER_mean']+b, color='red', label='Regression Line')
plt.title('Correlation between Sentiment and Bitcoin Price')
plt.xlabel('Sentiment (VADER_mean)')
plt.ylabel('Bitcoin Price (close)')
plt.show()