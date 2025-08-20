import pandas as pd
import numpy as np
import backtrader as bt
import load_data, feature_engineering, model_components
LOT_SIZE = 20000

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

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        if self.order:
            return
        
        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        position_value = portfolio_value - cash

        self.log(f"Portfolio: ${portfolio_value:.2f}, Cash: ${cash:.2f}, Position: ${position_value:.2f}")

        prediction = self.predictions[0]

        if prediction == 2:
            self.order = self.order_target_size(target=LOT_SIZE)
        elif prediction == 1:
            self.order = self.order_target_size(target=-LOT_SIZE)
        else:
            self.order = self.order_target_size(target=0)


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

def backtest(data, prune_length, macro_predictions, tech_predictions):
    data_frame = data[len(data) - prune_length:][["date", "Open", "Close", "High", "Low", "Change"]].copy()
    if len(macro_predictions) < len(tech_predictions):
        macro_predictions = macro_predictions[len(macro_predictions) - prune_length:]
    else:
        tech_predictions = tech_predictions[len(tech_predictions) - prune_length:]

    data_frame["date"] = pd.to_datetime(data_frame["date"]) 
    data_frame = data_frame.set_index("date")
    #data_frame.reset_index(inplace=True)
    data_frame["predictions"] = model_components.hybrid_predict(macro_predictions, tech_predictions)
    data_frame = data_frame.rename(columns={"Open": "open", "Close": "close", "High":"high", "Low": "low", "Change" : "truth_predictions"})
    #data_frame["truth_predictions"] = data_frame["truth_predictions"].shift(-1)
    data.dropna(inplace=True)

    test_data = TradingData(dataname=data_frame)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Strategy)
    cerebro.adddata(test_data)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00003, commtype=bt.CommInfoBase.COMM_PERC) #Based on IC markets commission 
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())


target_column, feature_columns_macro, feature_columns_tech = load_data.get_features_and_targets()
data = load_data.load()
sequence_length_tech, sequence_length_macro = 50, 80
features_test_macro, features_train_macro, targets_test_macro, targets_train_macro, test_dates_macro, scaler_macro = feature_engineering.split_data(data.copy(deep = True), feature_columns_macro, target_column, sequence_length_macro)
features_test_tech, features_train_tech, targets_test_tech, targets_train_tech, test_dates_tech, scaler_tech = feature_engineering.split_data(data, feature_columns_tech, target_column, sequence_length_tech)
macro_model, _ = model_components.train_model(sequence_length_macro, feature_columns_macro, features_train_macro, targets_train_macro,"macro_model")
tech_model, _ = model_components.train_model(sequence_length_tech, feature_columns_tech, features_train_tech, targets_train_tech, "tech_model")
macro_predictions = macro_model.predict(features_test_macro)
tech_predictions = tech_model.predict(features_test_tech)

backtest(data, min(len(test_dates_tech), len(test_dates_macro)), macro_predictions, tech_predictions)
