from fastapi import FastAPI
from backtest import backtest

app = FastAPI()

@app.get("/backtest")
def run_backtest():#start_date: str, end_date: str, leverage: int, start_cash: float
    return {"equity_curve" : []}
