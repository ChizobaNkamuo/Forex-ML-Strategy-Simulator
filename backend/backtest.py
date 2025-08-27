import pandas as pd
import numpy as np
import backtrader as bt
import load_data, feature_engineering, model_components, joblib, os
LOT_SIZE = 800000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

class Strategy(bt.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.data_close = self.datas[0].close
        self.predictions = self.datas[0].predictions
        #self.truth_predictions = self.datas[0].truth_predictions
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm {order.executed.comm}")

            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm {order.executed.comm}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        if trade.pnl < 0:
            self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" %
                 (trade.pnl, trade.pnlcomm))

    def next(self):        
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        position_value = portfolio_value - cash
        current_date = self.datas[0].datetime.date(0)
        prediction = self.predictions[0]
        price = self.data_close[0]
        self.log(f"Portfolio: ${portfolio_value:.2f}, Cash: ${cash:.2f}, Position: ${position_value:.2f}")
        current_position = self.broker.getposition(self.data).size
        
        if ((prediction == 2 and current_position < 0) or   # Want long but short
            (prediction == 1 and current_position > 0) or   # Want short but long  
            (prediction == 0 and current_position != 0)):   # Want flat but have position
            self.close()
            if self.stop_order:
                self.cancel(self.stop_order)
            
        if prediction == 2 and current_position <= 0:
            order = self.buy(size=LOT_SIZE, transmit=False)
            self.stop_order = self.sell(
                exectype=bt.Order.Stop, 
                price= price * 0.98,
                parent=order
            )

        elif prediction == 1 and current_position >= 0:
            order = self.sell(size=LOT_SIZE, transmit=False)
            self.stop_order = self.buy(
                exectype=bt.Order.Stop, 
                price= price * 1.02,
                parent=order
            )
    


class TradingData(bt.feeds.PandasData):
    lines = ("predictions",)
    params = (
        ("datetime", None),
        ("volume", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("predictions", -1), 
        #("truth_predictions", -1),
    )

def backtest(start_date = "2024-01-26", end_date = "2025-08-04", leverage = 10, start_cash = 100000.0):
    target_column, feature_columns_macro, feature_columns_tech = load_data.get_features_and_targets()
    data = load_data.load()
    data, _ = feature_engineering.add_indicators(data)
    data = data[data["date"] >= start_date]
    data = data[data["date"] <= end_date]    
    sequence_length_tech, sequence_length_macro = 90, 20
    #features_test_macro, features_train_macro, targets_test_macro, targets_train_macro, test_dates_macro, scaler_macro = feature_engineering.split_data(data.copy(deep = True), feature_columns_macro, target_column, sequence_length_macro)
    #features_test_tech, features_train_tech, targets_test_tech, targets_train_tech, test_dates_tech, scaler_tech = feature_engineering.split_data(data, feature_columns_tech, target_column, sequence_length_tech)
    
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
    print(data)
    
    data_frame = data[len(data) - prune_length:][["date", "Open", "Close", "High", "Low", "Change"]].copy()
    if len(tech_predictions) < len(macro_predictions):
        macro_predictions = macro_predictions[len(macro_predictions) - prune_length:]
    else:
        tech_predictions = tech_predictions[len(tech_predictions) - prune_length:]
        
    data_frame["date"] = pd.to_datetime(data_frame["date"]) 
    data_frame = data_frame.set_index("date")
    #data_frame.reset_index(inplace=True)
    data_frame["predictions"] = model_components.hybrid_predict(macro_predictions, tech_predictions)#data_frame["Change"].shift(-1)#
    data_frame = data_frame.rename(columns={"Open": "open", "Close": "close", "High":"high", "Low": "low"})
    #data_frame["truth_predictions"] = data_frame["truth_predictions"].shift(-1)
    data.dropna(inplace=True)

    test_data = TradingData(dataname=data_frame)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.adddata(test_data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="returns")
    
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.00003, commtype=bt.CommInfoBase.COMM_PERC, leverage = leverage) #Based on IC markets commission 
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.DrawDown)
    results = cerebro.run()[0]
    drawdown = results.analyzers.drawdown.get_analysis()

    print("Max Drawdown: %.2f%%" % drawdown.max.drawdown)
    print("Max Drawdown Money: %.2f" % drawdown.max.moneydown)
    print("Max Drawdown Length: %d" % drawdown.max.len)
    print("Sharpe Ratio:",  results.analyzers.sharpe.get_analysis()["sharperatio"])
    print("Returns: ", results.analyzers.returns.get_analysis())
    print(results.observers.broker.lines.value)
    equity_curve = list(results.observers.broker.lines.value)
    #drawdown_data = list(results.observers.drawdown.lines.value)

    return {"equity_curve": equity_curve}