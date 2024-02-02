import ccxt
import pandas as pd


def fetch_crypto_data(symbol, start_date, end_date, timeframe, exchange='binance'):
    exchange = getattr(ccxt, exchange)()
    start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=start_timestamp, limit=None, params={'endTime': end_timestamp})

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'date'})
    # df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

    # Save the DataFrame to a CSV file
    csv_filename = f'{symbol.replace("/", "_")}_data.csv'
    df.to_csv(csv_filename, index=False)
    return df