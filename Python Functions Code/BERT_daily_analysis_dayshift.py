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
Reddit_data = pd.read_csv('sentiment_all_subreddits_data.csv')

# Group and aggregate data
Reddit_data['comment_date'] = pd.to_datetime(Reddit_data['comment_date'])
Reddit_data['comment_date'] = Reddit_data['comment_date'].dt.strftime('%Y-%m-%d')
Reddit_data = Reddit_data.groupby('comment_date').agg(BERT_label=('BERT_label', 'mean')).reset_index()

# Second to Run the collection of Cryptocurrency price action
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y.%m.%d')
end_date = (datetime.now()).strftime('%Y.%m.%d')
symbol = 'BTC/USDT'
timeframe = '1d'
crypto_data = fetch_crypto_data(symbol, start_date, end_date, timeframe)


# Load Cryptocurrency price data
price_data = pd.read_csv(f'{symbol.replace("/", "_")}_data.csv')

#Update the prices to include the dayshift of 1 day. Code moves the price one row upwards
price_data['shift_close'] = price_data['close'].shift(-1)

# New column that shows the real date of the close value
price_data['date'] = pd.to_datetime(price_data['date'])
price_data['shift_date'] = price_data['date'] + pd.to_timedelta(1, unit='D')
price_data['shift_date'] = price_data['shift_date'].dt.strftime('%Y.%m.%d')

# Merge data on the common column ('comment_date' and 'timestamp')
merged_data = pd.merge(Reddit_data, price_data, left_on='comment_date', right_on='date', how='inner')

# Save the DataFrame to a CSV file
csv_filename = f'merged_data.csv'
merged_data.to_csv(csv_filename, index=False)

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation_volume = round(merged_data['BERT_label'].corr(merged_data['volume']),3)
print(f"Correlation between sentiment and Bitcoin Trading Volume: {correlation_volume}")

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation_price = round(merged_data['BERT_label'].corr(merged_data['close']),3)
print(f"Correlation between sentiment and Bitcoin price: {correlation_price}")

# Plot the correlation
plt.scatter(merged_data['BERT_label'], merged_data['volume'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['BERT_label'], merged_data['volume'], 1)
plt.plot(merged_data['BERT_label'], m*merged_data['BERT_label']+b, color='red', label='Regression Line')
plt.title(f'Correlation {correlation_volume} between Sentiment and Trading Volume')
plt.xlabel('Sentiment (BERT_label)')
plt.ylabel('Bitcoin Price (close)')
plt.show()

# Plot the correlation
plt.scatter(merged_data['BERT_label'], merged_data['close'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['BERT_label'], merged_data['close'], 1)
plt.plot(merged_data['BERT_label'], m*merged_data['BERT_label']+b, color='red', label='Regression Line')
plt.title(f'Correlation {correlation_price} between Sentiment and Bitcoin Price')
plt.xlabel('Sentiment (BERT_label)')
plt.ylabel('Bitcoin Price (close)')
plt.show()

# Plotting Trading Volume Data
plt.figure(figsize=(8, 5))
plt.plot(merged_data['comment_date'], merged_data['volume'], label='volume')
plt.title('BitCoin Trading Volume Data')
plt.xlabel('Date')
plt.ylabel('Trading volume')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Bitcoin Close Price Per date
plt.figure(figsize=(8, 5))
plt.plot(merged_data['comment_date'], merged_data['close'], label='Close')
plt.title('BitCoin Price Data')
plt.xlabel('Date')
plt.ylabel('Trading volume')
plt.legend()
plt.grid(True)
plt.show()
