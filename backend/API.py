from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backtest import backtest

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/backtest")
def run_backtest(start_date: str, end_date: str, leverage: int, units: float, start_cash: float):#leverage: int, start_cash: float
    print(start_date, end_date)
    return backtest(start_date = start_date, end_date = end_date, leverage = leverage, units = round(units), start_cash = start_cash)
