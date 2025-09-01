import pandas as pd
import numpy as np
import backtrader as bt
import load_data, feature_engineering, model_components, joblib, os
UNITS = 800000 #Default number of units for trades
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

class Strategy(bt.Strategy):
    params = (
        ("units", UNITS),
    )

    def log(self, txt):
        """
        Adds logged events to a list to be returned to the user
        """
        dt = self.datas[0].datetime.date(0).strftime("%d-%m-%Y")
        self.trades.append(f"{dt}, {txt}")

    def __init__(self):
        self.data_close = self.datas[0].close
        self.predictions = self.datas[0].predictions
        self.order = None
        self.trades = []

    def notify_order(self, order):
        """
        Logs the status of orders
        """
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price}, Comm {round(order.executed.comm,2)}")

            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price}, Comm {round(order.executed.comm, 2)}")

        elif order.status in [order.Margin]:
            self.log("ORDER REJECTED - Insufficient Margin")

        self.order = None

    def next(self):     
        """
        If the model predicts differently from it's last prediction -> Close any open positions
        And then if the new prediction isn't do nothing -> Open a new position
        """   
        prediction = self.predictions[0]
        price = self.data_close[0]
        current_position = self.broker.getposition(self.data).size
        
        if ((prediction == 2 and current_position < 0) or   # Want long but short
            (prediction == 1 and current_position > 0) or   # Want short but long  
            (prediction == 0 and current_position != 0)):   # Want flat but have position
            self.close()
            if self.stop_order:
                self.cancel(self.stop_order)
            
        if prediction == 2 and current_position <= 0:
            order = self.buy(size=self.p.units, transmit=False)
            self.stop_order = self.sell( #Set stop loss of 2%
                exectype=bt.Order.Stop, 
                price= price * 0.98,
                parent=order
            )

        elif prediction == 1 and current_position >= 0:
            order = self.sell(size=self.p.units, transmit=False)
            self.stop_order = self.buy( #Set stop loss of 2%
                exectype=bt.Order.Stop, 
                price= price * 1.02,
                parent=order
            )
    


class TradingData(bt.feeds.PandasData):
    lines = ("predictions",) #Add a custom column to the data - predictions
    params = (
        ("datetime", None),
        ("volume", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("predictions", -1), 
    )

def backtest(start_date, end_date, leverage, start_cash, units):
    """
    Loads pre-trained models and runs a backtest simulation using the specified parameters
    """
    target_column, feature_columns_macro, feature_columns_tech = load_data.get_features_and_targets()
    data = load_data.load()
    sequence_length_tech, sequence_length_macro = 90, 20
    data, _ = feature_engineering.add_indicators(data)
    data = data.reset_index(drop=True)
    start_position = data[data["date"] >= pd.to_datetime(start_date)].index[0]
    required_lookback = max(sequence_length_tech, sequence_length_macro)
    data = data.iloc[max(0, start_position - required_lookback):]
    data = data[data["date"] <= pd.to_datetime(end_date)]    

    features_test_macro, features_test_tech = data[feature_columns_macro], data[feature_columns_tech]
    targets_test = data[target_column]

    tech_scaler = joblib.load(os.path.join(MODEL_DIR, "tech_scaler.gz"))
    macro_scaler = joblib.load(os.path.join(MODEL_DIR, "macro_scaler.gz"))

    features_test_macro, features_test_tech = macro_scaler.transform(features_test_macro), tech_scaler.transform(features_test_tech)
    features_test_macro, _ = feature_engineering.create_sequences(features_test_macro, targets_test.values, sequence_length_macro)
    features_test_tech, _ = feature_engineering.create_sequences(features_test_tech, targets_test.values, sequence_length_tech)  

    macro_model = model_components.train_macro_model("macro_model", sequence_length_macro, feature_columns_macro)
    tech_model = model_components.train_tech_model("tech_model", sequence_length_tech, feature_columns_tech)
    macro_predictions = macro_model.predict(features_test_macro)
    tech_predictions = tech_model.predict(features_test_tech)
    prune_length = min(len(features_test_macro), len(features_test_tech))
    data_frame = data[data["date"] >= pd.to_datetime(start_date)][["date", "Open", "Close", "High", "Low", "Change"]].copy()

    if len(tech_predictions) < len(macro_predictions):
        macro_predictions = macro_predictions[len(macro_predictions) - prune_length:]
    else:
        tech_predictions = tech_predictions[len(tech_predictions) - prune_length:]
    data_frame = data_frame.set_index("date")
    #data_frame.reset_index(inplace=True)
    data_frame["predictions"] = model_components.hybrid_predict(macro_predictions, tech_predictions)#data_frame["Change"].shift(-1)#
    data_frame = data_frame.rename(columns={"Open": "open", "Close": "close", "High":"high", "Low": "low"})
    #data_frame["truth_predictions"] = data_frame["truth_predictions"].shift(-1)
    data.dropna(inplace=True)

    test_data = TradingData(dataname=data_frame)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy, units = units)
    cerebro.adddata(test_data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalyzer")
    
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.00003, commtype=bt.CommInfoBase.COMM_PERC, leverage = leverage) #Based on IC markets commission 
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.DrawDown)
    
    results = cerebro.run()[0]
    dates = cerebro.datas[0].p.dataname.index.strftime("%Y-%m-%d").tolist()
    equity_values = np.array(results.observers.broker.lines.value.array)
    equity_values = equity_values[~np.isnan(equity_values)].tolist()

    equity_curve = [{"time" : date, "value": value} for date,value in zip(dates, equity_values)]
    candle_sticks = [{"open": open, "high": high, "low": low, "close": close, "time": date}  
                     for open, high, low, close, date in zip(data_frame["open"].values, data_frame["high"].values, data_frame["low"].values, data_frame["close"].values, dates)]
    markers = [{"time": date, "size": 0.01,"position": "aboveBar", "shape": "arrowUp" if prediction == 2 else "arrowDown", 
                "color": "rgba(34, 197, 94, 0.5)" if prediction == 2 else "rgba(239, 68, 68, 0.5)"} for date, prediction in zip(dates, data_frame["predictions"].values) if prediction > 0]

    return {"equity_curve": equity_curve, "candle_sticks": candle_sticks, "markers": markers, "stats": generate_metrics(cerebro, results, data_frame), "trades" : results.trades}

def generate_metrics(cerebro, results, data_frame):
    """
    Generates useful metrics from the simulation results
    """
    drawdown = results.analyzers.drawdown.get_analysis()
    returns_analysis = results.analyzers.returns.get_analysis()#["rtot"]
    trade_analysis = results.analyzers.tradeanalyzer.get_analysis()

    total_trades = trade_analysis.get("total", {}).get("total", 0)
    win_trades = trade_analysis.get("won", {}).get("total", 0)
    win_rate = win_trades / total_trades if total_trades != 0 else 0

    gross_profit = trade_analysis.get("won", {}).get("pnl", {}).get("total", 0)
    gross_loss   = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("total", 0))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else None

    largest_win = trade_analysis.get("won", {}).get("pnl", {}).get("max", 0)
    largest_loss = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("max", 0) )
    avg_win = trade_analysis.get("won", {}).get("pnl", {}).get("average", 0)
    avg_loss = abs(trade_analysis.get("lost", {}).get("pnl", {}).get("average", 0))
    pay_off_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None

    returns_percent = 0

    for percent in returns_analysis.values():
        returns_percent += percent

    sharpe_ratio = results.analyzers.sharpe.get_analysis()["sharperatio"]

    return {
        "Sharpe Ratio" : round(sharpe_ratio, 2) if sharpe_ratio else "N/A",
        "Max Drawdown" : str(round(drawdown.max.drawdown, 2)) + "%",
        "Total Returns": str(round(returns_percent * 100, 2)) + "%",
        "Total Trades" : total_trades,
        "Win Rate" : str(round(win_rate * 100, 2)) + "%",
        "Payoff Ratio" : round(pay_off_ratio, 2) if pay_off_ratio else "N/A",
        "Profit Factor" : round(profit_factor, 2) if profit_factor else "N/A",
        "Largest Win" : "$"+ str(round(largest_win, 2)),
        "Largest Loss" : "$"+ str(round(largest_loss, 2)),
        "Average Win" : "$"+ str(round(avg_win, 2)),
        "Average Loss" : "$"+ str(round(avg_loss, 2)),
    }
