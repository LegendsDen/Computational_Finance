import os.path
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt


# Basic I/O functions
def inp(): return int(sys.stdin.readline())
def st(): return list(sys.stdin.readline().strip())
def li(): return list(map(int, sys.stdin.readline().split()))
def mp(): return map(int, sys.stdin.readline().split())
def pr(n): return sys.stdout.write(str(n) + "\n")
def prl(n): return sys.stdout.write(str(n) + "")

# Redirect input/output for debugging with files if they exist
if os.path.exists('input.txt'):
    sys.stdin = open('input.txt', 'r')
    sys.stdout = open('output.txt', 'w')

# Settings for data retrieval
strike_price=1750;
start_date = "2024-09-25"
end_date = "2024-10-25"
expiry_date = pd.to_datetime("2024-10-31")
default_r = 0.05  # Default risk-free rate if data is unavailable
expiry_date = expiry_date.tz_localize('UTC')
print(expiry_date)


tickers_strikes = [
    {"ticker": "HDFCBANK.NS", "file_path": "C:/Python/Quote-FAO-HDFCBANK-27-09-2024-to-27-10-2024.csv"},
]


# Black-Scholes function
def Black_Scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

# Binomial Model function
def Binomial_Model(S, K, T, r, mu, sigma, steps, option_type="call"):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = 0.5 + (mu * np.sqrt(dt)) / (2 * sigma)

    prices = np.zeros(steps + 1)
    prices[0] = S * (d ** steps)
    for i in range(1, steps + 1):
        prices[i] = prices[i - 1] * (u / d)

    values = np.maximum(0, (prices - K) if option_type == "call" else (K - prices))
    discount = np.exp(-r * dt)
    for step in range(steps - 1, -1, -1):
        values = discount * (q * values[1:step + 2] + (1 - q) * values[:step + 1])

    return values[0]

# Main process
for item in tickers_strikes:
    ticker = item["ticker"]
    file_path = item["file_path"]

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    stock_data.dropna(inplace=True)
    # print(stock_data)
    # Calculate daily returns
    stock_data['Return'] = stock_data['Adj Close'].pct_change()

    # Calculate sigma and mu
    historical_data = yf.Ticker(ticker).history(period='1y')
    historical_data.dropna(inplace=True)
    sigma = historical_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
    mu = historical_data['Close'].pct_change().mean() * 252            # Annualized mean return

    # Download risk-free rate data (T-bill rate) and prepare for merging
    risk_free_data = yf.download("^IRX", start=start_date, end=end_date, interval="1d", progress=False)
    risk_free_data.dropna(inplace=True)
    risk_free_data = risk_free_data[['Close']].rename(columns={'Close': 'risk_free_rate'})  # Rename Close to risk_free_rate

    # Merge stock data with risk-free rate
    stock_data = stock_data.merge(risk_free_data, left_index=True, right_index=True, how='left')
    stock_data['risk_free_rate'] = stock_data['risk_free_rate'].fillna(default_r) / 100  # Fill NaN and convert to decimal

    # print(stock_data)


    option_prices = []


    for current_date, row in stock_data.iterrows():
        # print(current_date)
        Stock_price = row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else float(row['Close'])
        Rate_of_interest = row['risk_free_rate'].iloc[0] if isinstance(row['risk_free_rate'], pd.Series) else float(row['risk_free_rate'])


        T = (expiry_date - current_date).days / 365
        if T < 0:
            option_price_bs = np.nan
            option_price_bin = np.nan
        else:
            Stock_price = float(Stock_price)
            option_price_bs = Black_Scholes(S=float(Stock_price), K=float(strike_price), T=float(T), r=float(Rate_of_interest), sigma=float(sigma), option_type='call')            # option_price_bin = binomial_model(S=S_t, K=strike_price, T=T, r=r_t,mu= mu, sigma=sigma, steps=10000, option_type='call')
            option_price_bin = Binomial_Model(S=float(Stock_price), K=float(strike_price), T=float(T), r=float(Rate_of_interest), mu=float(mu), sigma=float(sigma), steps=10000, option_type='call')

        option_prices.append({
            'Date': current_date,
            'Stock_Price': Stock_price,
            'Black_Scholes_Call_Price': option_price_bs,
            'Binomial_Model_Call_Price': option_price_bin
        })


    option_prices_df = pd.DataFrame(option_prices)
    option_prices_df.set_index('Date', inplace=True)


    HDFCBANK_actual_prices = pd.read_csv(file_path)

    HDFCBANK_actual_prices.columns = HDFCBANK_actual_prices.columns.str.strip()
    HDFCBANK_actual_prices['DATE'] = pd.to_datetime(HDFCBANK_actual_prices['DATE'], format='%d-%b-%Y')


    actual_option_prices = HDFCBANK_actual_prices[['DATE', 'CLOSE PRICE']]


    option_prices_df = option_prices_df.reset_index()
    option_prices_df['Date'] = option_prices_df['Date'].dt.tz_localize(None)


    option_prices_df = option_prices_df.merge(actual_option_prices, left_on='Date', right_on='DATE', how='left')


    final_columns = ['Date', 'Stock_Price',
                     'Black_Scholes_Call_Price', 'Binomial_Model_Call_Price', 'CLOSE PRICE']

    print(f"\n{ticker} - Predicted vs Actual Option Prices:")
    print(option_prices_df[final_columns])
    plt.figure(figsize=(14, 7))
    plt.plot(option_prices_df['Date'], option_prices_df['Black_Scholes_Call_Price'], label='Black-Scholes Call Price', marker='o')
    plt.plot(option_prices_df['Date'], option_prices_df['Binomial_Model_Call_Price'], label='Binomial Model Call Price', marker='x')
    plt.plot(option_prices_df['Date'], option_prices_df['CLOSE PRICE'], label='Actual Close Price', linestyle='--', marker='s')
    plt.xlabel('Date')
    plt.ylabel('Option Price')
    plt.title(f'Call Option Price: Black-Scholes, Binomial, and Actual for {ticker}')
    plt.legend()
    plt.grid(True)
    plt.show()
