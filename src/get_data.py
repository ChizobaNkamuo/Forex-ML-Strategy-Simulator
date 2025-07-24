import yfinance as yf
import pandas as pd

def get_data(interval = "1d", pair = "EURUSD"):
    start_date = "2023-07-30"
    end_date = "2025-07-24"

    forex_data = yf.download(f"{pair}=X", start=start_date, end=end_date, interval = interval)
    forex_data.index = pd.to_datetime(forex_data.index)
    forex_data.to_csv(f"{pair}{start_date}-to-{end_date}-{interval}.csv", index=False)

get_data("1d")