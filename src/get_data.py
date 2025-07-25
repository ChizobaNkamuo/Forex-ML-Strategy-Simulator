import yfinance as yf
import pandas as pd

def get_data(interval = "1d", pair = "EURUSD"):
    start_date = "2021-07-29"
    end_date = "2025-07-01"
    #2021-07-29-2025-06-01
    forex_data = yf.download(f"{pair}=X", start=start_date, end=end_date, interval = interval)
    forex_data = forex_data.reset_index()
    forex_data = forex_data.rename(columns={"Date": "date"})
    forex_data.to_csv(f"{pair}{start_date}-to-{end_date}-{interval}.csv", index=False)

get_data("1d")
