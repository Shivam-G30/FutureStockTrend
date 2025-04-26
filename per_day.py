import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Configuration
TICKER = "MRF.NS"  # Stock ticker (State Bank of India, NSE)
HISTORY_YEARS = 4   # Fetch past 4 years of data
FORECAST_DAYS = 30  # Forecast next 30 business days

def fetch_stock_data(ticker):
    """Fetch historical stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
    print(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}")
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

def calculate_technical_indicators(df):
    """Calculate Moving Averages, RSI, MACD"""
    df = df.copy()

    # Moving Averages
    df['50_MA'] = df['Close'].rolling(50).mean()
    df['200_MA'] = df['Close'].rolling(200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def generate_forecast(df, forecast_days=30):
    """Forecast future prices using Linear Regression"""
    prices = df['Close'].values.reshape(-1, 1)
    days = np.arange(len(prices)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(days, prices)

    future_days = np.arange(len(prices), len(prices) + forecast_days).reshape(-1, 1)
    future_prices = model.predict(future_days)

    last_date = df.index[-1]
    date_range = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

    return pd.DataFrame({'Date': date_range, 'Forecast': future_prices.flatten()}).set_index('Date')

def plot_analysis(history, forecast):
    """Plot Historical and Forecasted Data"""
    plt.figure(figsize=(15, 12))

    # 1. Price and Forecast
    plt.subplot(3, 1, 1)
    plt.plot(history.index, history['Close'], label='Historical Price', color='blue')
    plt.plot(forecast.index, forecast['Forecast'], label='Forecast (Next 30 Days)', color='red', linestyle='--')
    plt.scatter(forecast.index, forecast['Forecast'], color='red', s=20)
    plt.title(f'{TICKER} Analysis (Last {HISTORY_YEARS} Years + Next 30 Days Forecast)')
    plt.ylabel('Price (₹)')
    plt.legend()

    # 2. MACD
    plt.subplot(3, 1, 2)
    plt.plot(history.index, history['MACD'], label='MACD', color='purple')
    plt.plot(history.index, history['Signal'], label='Signal', color='orange')
    plt.title('MACD Indicator')
    plt.legend()

    # 3. RSI
    plt.subplot(3, 1, 3)
    plt.plot(history.index, history['RSI'], label='RSI', color='green')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title('RSI Indicator')
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_forecast(forecast):
    """Print daily forecast for next 30 business days"""
    if forecast is None or forecast.empty:
        print("No forecast available.")
        return

    print("\n=== Next 30 Business Days Forecast ===")
    for date, row in forecast.iterrows():
        print(f"{date.strftime('%d-%b-%Y')} : ₹{row['Forecast']:.2f}")

    # Optionally save to CSV
    forecast.to_csv('daily_forecast.csv')
    print("\nForecast also saved to 'daily_forecast.csv'.")

def main():
    """Main Program Execution"""
    try:
        stock_data = fetch_stock_data(TICKER)
        if stock_data.empty:
            raise ValueError("No data fetched. Check Ticker Symbol or Internet Connection.")

        df = calculate_technical_indicators(stock_data)
        forecast = generate_forecast(df, FORECAST_DAYS)

        # Print Current Status
        current_price = float(stock_data['Close'].iloc[-1])
        print(f"\nCurrent {TICKER} Price: ₹{current_price:.2f}")
        print(f"RSI: {df['RSI'].iloc[-1]:.1f}")
        print(f"MACD Signal: {'Bullish' if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else 'Bearish'}")

        print_forecast(forecast)
        plot_analysis(df, forecast)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Possible fixes:")
        print("1. Verify the ticker symbol (e.g., use '.NS' for Indian stocks).")
        print("2. Ensure your internet connection is active.")
        print("3. Try running during market days/hours.")

if __name__ == "__main__":
    main()
