import pandas as pd
import numpy as np
import backtrader as bt

class TradingData(bt.feeds.PandasData):
    lines = ("open_prediction", "close_prediction", "high_prediction", "low_prediction", "direction") 
    params = (
        ("volume", -1), 
        ("open_prediction", -1), 
        ("close_prediction", -1),
        ("high_prediction", -1),
        ("low_prediction", -1),
        ("direction", -1)
    )

def backtest(test_data, dates):
    #print(test_data)
    #test_data = np.append(test_data[:, :4].squeeze(), dates.values[:, np.newaxis], axis=1)
    data_frame = pd.DataFrame(test_data[:, :4].squeeze(), columns=["open", "close", "high", "low"])
    data_frame["datetime"] = dates.values
    data_frame.set_index("datetime")

    test_data = TradingData(dataname=data_frame)
    #Add a date and finish class