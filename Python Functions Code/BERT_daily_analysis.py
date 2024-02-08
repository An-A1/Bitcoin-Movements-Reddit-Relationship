import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from Reddit_data_Fun import *
from Cryptocurrency_Fun import *
from VADER_Sentiment import *
from BERT_Sentiment import *



# First to Run the collection of Rediit Data
#get_reddit_data(subreddit_list, limit=30, csv_file_path=csv_path, filter_words=filter_words_list)

#Call the function to Perform the sentiment analysis in the Redditthe DataFrame
# BERT_sentiments(input_csv='Reddit_new.csv', output_csv='BERT_sentiment_data.csv')

#Load Reddit Data for perform the merge and Corrolation
Reddit_data = pd.read_csv('BERT_sentiment_data.csv')

# Group and aggregate data
Reddit_data['comment_date'] = pd.to_datetime(Reddit_data['comment_date'])
Reddit_data['comment_date'] = Reddit_data['comment_date'].dt.strftime('%Y-%m-%d')
Reddit_data = Reddit_data.groupby('comment_date').agg(BERT_Compound=('BERT_Compound', 'mean')).reset_index()

# Second to Run the collection of Cryptocurrency price action
start_date = (datetime.now() - timedelta(days=50)).strftime('%Y.%m.%d')
end_date = (datetime.now()).strftime('%Y.%m.%d')
symbol = 'BTC/USDT'
timeframe = '1d'
crypto_data = fetch_crypto_data(symbol, start_date, end_date, timeframe)


# Load Cryptocurrency price data
price_data = pd.read_csv(f'{symbol.replace("/", "_")}_data.csv')


# Merge data on the common column ('comment_date' and 'timestamp')
merged_data = pd.merge(Reddit_data, price_data, left_on='comment_date', right_on='date', how='inner')
# merged_data['BERT_shifted_up'] = merged_data['BERT_Compound'].shift(-7)
# merged_data['date_shifted_up'] = merged_data['comment_date'].shift(-7)
# merged_data['BERT_shifted_down'] = merged_data['BERT_Compound'].shift(+7)
# merged_data['date_shifted_down'] = merged_data['comment_date'].shift(+7)
# merged_data = merged_data.dropna()
csv_filename = f'merged_data.csv'
merged_data.to_csv(csv_filename, index=False)


# Plotting Bitcoin Close Price Per date
plt.figure(figsize=(10, 5))
plt.plot(merged_data['comment_date'], merged_data['close'], label='Close')
plt.title('BitCoin Price Data')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xlabel('Date: 1st - 29th January, 2024')
plt.ylabel('BTC Closig Price')
plt.legend()
plt.grid(True)
plt.show()

# Plotting BERT Analysis data over date
plt.figure(figsize=(10, 5))
plt.plot(merged_data['comment_date'], merged_data['BERT_Compound'], label='BERT_Compound')
plt.title('BERT Sentiment Analysis data')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xlabel('Date: 1st - 29th January, 2024')
plt.ylabel('BERT Compound Score')
plt.legend()
plt.grid(True)
plt.show()

# Create a histogram Histogram of BERT_Compound Scores
mean_value = merged_data['BERT_Compound'].mean()
standard_deviation = np.std(merged_data['BERT_Compound'], ddof=1)
plt.hist(merged_data['BERT_Compound'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(mean_value + standard_deviation, color='green', linestyle='dashed', linewidth=2, label='Std Dev')
plt.axvline(mean_value - standard_deviation, color='green', linestyle='dashed', linewidth=2)
plt.xlabel('BERT Compound Score')
plt.ylabel('Frequency')
plt.title('BERT sentiment Compound Scores Mean')
plt.legend()
plt.show()


# Calculate correlation between Reddit Sentiment Data and Cryptocurrency volume
correlation_volume = round(merged_data['BERT_Compound'].corr(merged_data['volume']),3)
print(f"Correlation between sentiment and Bitcoin Trading Volume: {correlation_volume}")
# Plot the correlation
plt.scatter(merged_data['BERT_Compound'], merged_data['volume'])
# obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['BERT_Compound'], merged_data['volume'], 1)
plt.plot(merged_data['BERT_Compound'], m*merged_data['BERT_Compound']+b, color='red', label='Regression Line')
plt.legend()
plt.title(f'Correlation {correlation_volume} between Sentiment and Trading Volume')
plt.xlabel('BERT sentiment compound scores')
plt.ylabel('Bitcoin closing price (USD)')
plt.show()

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation_price = round(merged_data['BERT_Compound'].corr(merged_data['close']),3)
print(f"Correlation between sentiment and Bitcoin price same day: {correlation_price}")
# Plot the correlation
plt.scatter(merged_data['BERT_Compound'], merged_data['close'])
# obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['BERT_Compound'], merged_data['close'], 1)
plt.plot(merged_data['BERT_Compound'], m*merged_data['BERT_Compound']+b, color='red', label='Regression Line')
plt.legend()
plt.title(f'Correlation {correlation_price} between Sentiment and Bitcoin Price')
plt.xlabel('BERT Sentiment compound score')
plt.ylabel('Bitcoin Closing Price (USD)')
plt.show()

# # Calculate correlation between Reddit Sentiment Data and Cryptocurrency price shifted up
# correlation_shifted_up = round(merged_data['BERT_shifted_up'].corr(merged_data['close']),3)
# print(f"Correlation between sentiment shifted up and Bitcoin price: {correlation_shifted_up}")
# # Plot the correlation
# plt.scatter(merged_data['BERT_shifted_up'], merged_data['close'])
# # obtain m (slope) and b(intercept) of linear regression line
# m, b = np.polyfit(merged_data['BERT_shifted_up'], merged_data['close'], 1)
# plt.plot(merged_data['BERT_shifted_up'], m*merged_data['BERT_shifted_up']+b, color='red', label='Regression Line')
# plt.legend()
# plt.title(f'Correlation {correlation_shifted_up} between Sentiment and Bitcoin Price')
# plt.xlabel('Sentiment (BERT_shifted_up)')
# plt.ylabel('Bitcoin Price (close)')
# plt.show()

# # Calculate correlation between Reddit Sentiment Data and Cryptocurrency price shifted down
# correlation_shifted_down = round(merged_data['BERT_shifted_down'].corr(merged_data['close']),3)
# print(f"Correlation between sentiment shifted down and Bitcoin price: {correlation_shifted_down}")
# # Plot the correlation
# plt.scatter(merged_data['BERT_shifted_down'], merged_data['close'])
# # obtain m (slope) and b(intercept) of linear regression line
# m, b = np.polyfit(merged_data['BERT_shifted_down'], merged_data['close'], 1)
# plt.plot(merged_data['BERT_shifted_down'], m*merged_data['BERT_shifted_down']+b, color='red', label='Regression Line')
# plt.legend()
# plt.title(f'Correlation {correlation_shifted_down} between Sentiment and Bitcoin Price')
# plt.xlabel('Sentiment (BERT_shifted_down)')
# plt.ylabel('Bitcoin Price (close)')
# plt.show()
