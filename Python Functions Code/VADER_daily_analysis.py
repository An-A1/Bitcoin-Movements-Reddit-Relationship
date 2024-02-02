import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from Reddit_data_Fun import *
from Cryptocurrency_Fun import *
from VADER_Sentiment import *



# First to Run the collection of Rediit Data
#get_reddit_data(subreddit_list, limit=30, csv_file_path=csv_path, filter_words=filter_words_list)

#Call the function to Perform the sentiment analysis in the Redditthe DataFrame
#VADER_sentiment_analysis(input_csv='Reddit_new.csv', output_csv='VADER_sentiment_data.csv')

#Load Reddit Data for perform the merge and Corrolation
Reddit_data = pd.read_csv('VADER_sentiment_data.csv')

# Group and aggregate data
Reddit_data['comment_date'] = pd.to_datetime(Reddit_data['comment_date'])
Reddit_data['comment_date'] = Reddit_data['comment_date'].dt.strftime('%Y-%m-%d')
Reddit_data = Reddit_data.groupby('comment_date').agg(VADER_mean=('VADER_compound', 'mean')).reset_index()

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
# merged_data['VADER_shifted_up'] = merged_data['VADER_mean'].shift(-5)
# merged_data['date_shifted_up'] = merged_data['comment_date'].shift(-5)
# merged_data['VADER_shifted_down'] = merged_data['VADER_mean'].shift(+5)
# merged_data['date_shifted_down'] = merged_data['comment_date'].shift(+5)
# merged_data = merged_data.dropna()
csv_filename = f'merged_data.csv'
merged_data.to_csv(csv_filename, index=False)

# Plotting Bitcoin Close Price Per date
plt.figure(figsize=(15, 5))
plt.plot(merged_data['comment_date'], merged_data['close'], label='Close Price', color = 'blue')
plt.title('BitCoin Price Over Date')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xlabel('Date as Jan 2024')
plt.ylabel('BTC closing Price')
plt.legend()
plt.grid(True)
plt.show()

# Plotting BERT Analysis data over date
plt.figure(figsize=(15, 5))
plt.plot(merged_data['comment_date'], merged_data['VADER_mean'], label='VADER_mean', color = 'green')
plt.title('VADER Sentiment daily average score')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xlabel('Date as Jan 2024')
plt.ylabel('VADER Analysis Score')
plt.legend()
plt.grid(True)
plt.show()

# Create a histogram Histogram of BERT_Compound Scores
mean_value = merged_data['VADER_mean'].mean()
standard_deviation = np.std(merged_data['VADER_mean'], ddof=1)
plt.hist(merged_data['VADER_mean'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(mean_value + standard_deviation, color='green', linestyle='dashed', linewidth=2, label='Std Dev')
plt.axvline(mean_value - standard_deviation, color='green', linestyle='dashed', linewidth=2)
plt.xlabel('VADER_mean Score')
plt.ylabel('Frequency')
plt.title('Histogram of VADER_mean Scores')
plt.legend()
plt.show()

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency Volume
correlation_volume = round(merged_data['VADER_mean'].corr(merged_data['volume']),3)
print(f"Correlation between sentiment and Bitcoin Trading Volume: {correlation_volume}")
# Plot the correlation
plt.scatter(merged_data['VADER_mean'], merged_data['volume'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['VADER_mean'], merged_data['volume'], 1)
plt.plot(merged_data['VADER_mean'], m*merged_data['VADER_mean']+b, color='red', label='Regression Line')
plt.title(f'Correlation {correlation_volume} between Sentiment and Trading Volume')
plt.xlabel('Sentiment (VADER_mean)')
plt.ylabel('Bitcoin Price (close)')
plt.show()

# Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
correlation_Price = round(merged_data['VADER_mean'].corr(merged_data['close']),3)
print(f"Correlation between sentiment and Bitcoin price: {correlation_Price}")
# Plot the correlation Sentiment vs Close
plt.scatter(merged_data['VADER_mean'], merged_data['close'])
#obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(merged_data['VADER_mean'], merged_data['close'], 1)
plt.plot(merged_data['VADER_mean'], m*merged_data['VADER_mean']+b, color='red', label='Regression Line')
plt.title(f'Correlation {correlation_Price} between Sentiment and Bitcoin Price')
plt.xlabel('Sentiment (VADER_mean)')
plt.ylabel('Bitcoin Price (close)')
plt.show()


# # Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
# correlation_vader_shifted_up = round(merged_data['VADER_shifted_up'].corr(merged_data['close']),3)
# print(f"Correlation between sentiment shifted up and Bitcoin price: {correlation_vader_shifted_up}")
# # Plot the correlation Sentiment vs Close shifted up
# plt.scatter(merged_data['VADER_shifted_up'], merged_data['close'])
# #obtain m (slope) and b(intercept) of linear regression line
# m, b = np.polyfit(merged_data['VADER_shifted_up'], merged_data['close'], 1)
# plt.plot(merged_data['VADER_shifted_up'], m*merged_data['VADER_shifted_up']+b, color='red', label='Regression Line')
# plt.title(f'Correlation {correlation_vader_shifted_up} between Sentiment and Bitcoin Price shifted up')
# plt.xlabel('Sentiment (VADER_shifted_up)')
# plt.ylabel('Bitcoin Price (close)')
# plt.show()


# # Calculate correlation between Reddit Sentiment Data and Cryptocurrency price
# correlation_vader_shifted_down = round(merged_data['VADER_shifted_down'].corr(merged_data['close']),3)
# print(f"Correlation between sentiment shifted down and Bitcoin price: {correlation_vader_shifted_down}")
# # Plot the correlation Sentiment vs Close shifted down
# plt.scatter(merged_data['VADER_shifted_down'], merged_data['close'])
# #obtain m (slope) and b(intercept) of linear regression line
# m, b = np.polyfit(merged_data['VADER_shifted_down'], merged_data['close'], 1)
# plt.plot(merged_data['VADER_shifted_down'], m*merged_data['VADER_shifted_down']+b, color='red', label='Regression Line')
# plt.title(f'Correlation {correlation_vader_shifted_down} between Sentiment and Bitcoin Price shifted down')
# plt.xlabel('Sentiment (VADER_shifted_down)')
# plt.ylabel('Bitcoin Price (close)')
