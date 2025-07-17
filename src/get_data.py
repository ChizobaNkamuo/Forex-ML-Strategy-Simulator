import yfinance as yf
import pandas as pd

start_date = "2020-01-01"
end_date = "2025-07-01"

forex_data = yf.download("EURUSD=X", start=start_date, end=end_date)
forex_data.index = pd.to_datetime(forex_data.index)
forex_data.to_csv("EURUSD2020-2025.csv", index=False)