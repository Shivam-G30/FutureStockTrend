from flask import Flask, render_template_string, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Default Config
HISTORY_YEARS = 4
FORECAST_MONTHS = 6

def fetch_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*HISTORY_YEARS)
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

def calculate_technical_indicators(df):
    df = df.copy()
    df['50_MA'] = df['Close'].rolling(50).mean()
    df['200_MA'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def generate_forecast(df, months=6):
    prices = df['Close'].values.reshape(-1, 1)
    days = np.array(range(len(prices))).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, prices)
    future_days = np.array(range(len(prices), len(prices)+30*months)).reshape(-1, 1)
    future_prices = model.predict(future_days)
    last_date = df.index[-1]
    date_range = pd.bdate_range(start=last_date + timedelta(days=1), periods=30*months)
    return pd.DataFrame({'Date': date_range, 'Forecast': future_prices.flatten()}).set_index('Date')

def plot_analysis(history, forecast, ticker):
    plt.figure(figsize=(15, 10))
    # Price
    plt.subplot(3, 1, 1)
    plt.plot(history.index, history['Close'], label='Historical Price', color='blue')
    if forecast is not None:
        plt.plot(forecast.index, forecast['Forecast'], label='Forecast', color='red', linestyle='--')
        monthly_forecast = forecast.resample('M').first()
        plt.scatter(monthly_forecast.index, monthly_forecast['Forecast'], color='red', s=100)
    plt.title(f'{ticker} Price Analysis')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 2)
    plt.plot(history.index, history['MACD'], label='MACD', color='purple')
    plt.plot(history.index, history['Signal'], label='Signal', color='orange')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 3)
    plt.plot(history.index, history['RSI'], label='RSI', color='green')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.legend()

    plt.tight_layout()
    if not os.path.exists('static'):
        os.makedirs('static')
    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ ticker }} Stock Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">

<div class="container py-5">
    <h1 class="mb-4">{{ ticker }} Stock Analysis</h1>

    <form method="post" class="mb-4">
        <input type="text" name="ticker" class="form-control" placeholder="Enter stock ticker (example: SBIN.NS)">
        <button type="submit" class="btn btn-primary mt-2">Analyze</button>
    </form>

    <div class="row mb-4">
        <div class="col">
            <h4>Current Price: ₹{{ price }}</h4>
            <h5>RSI: {{ rsi }}</h5>
            <h5>MACD Signal: {{ macd_signal }}</h5>
        </div>
    </div>

    <img src="/{{ chart_url }}" alt="Chart" class="img-fluid rounded shadow">

</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    ticker = request.form.get('ticker') or 'SBIN.NS'
    try:
        stock_data = fetch_stock_data(ticker)
        df = calculate_technical_indicators(stock_data)
        forecast = generate_forecast(df, FORECAST_MONTHS)
        chart_url = plot_analysis(df, forecast, ticker)

        current_price = round(stock_data['Close'].iloc[-1], 2)
        current_rsi = round(df['RSI'].iloc[-1], 2)
        macd_signal = "Bullish" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "Bearish"

        return render_template_string(HTML_TEMPLATE, ticker=ticker, price=current_price,
                                      rsi=current_rsi, macd_signal=macd_signal, chart_url=chart_url)
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
