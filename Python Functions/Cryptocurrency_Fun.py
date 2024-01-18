import ccxt
import pandas as pd
import matplotlib.pyplot as plt



def fetch_crypto_data(symbol, start_date, end_date, timeframe, exchange='binance'):
    # Instantiate the ccxt client for the specified exchange
    exchange = getattr(ccxt, exchange)()

    # Convert start and end dates to timestamps
    start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)

# Fetch OHLCV data for the specified symbol and time range
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=start_timestamp, limit=None, params={'endTime': end_timestamp})

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Extract the date and hour from the timestamp
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')


    # Save the DataFrame to a CSV file
    csv_filename = f'{symbol.replace("/", "_")}_data.csv'
    df.to_csv(csv_filename, index=False)
    return df


def plot_closing_prices(data, symbol):
    # Plot the close values over time
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamp'], data['close'], label='Close')
    plt.title(f'{symbol} Close Prices Over Time')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()
