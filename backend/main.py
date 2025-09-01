from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backtest import backtest

app = FastAPI()
"""
Configure CORS to allow requests 
from the local frontend during development and from any origin when deployed
"""

origins = [
    "http://localhost:5173", #Dev server
    "http://127.0.0.1:5173", #Alternate local host
    "*" #Allow all origins for deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def root(): #Add a root for elastic beanstalk health checks
    return {"message": "FastAPI app is running"}

@app.get("/backtest")
def run_backtest(start_date: str, end_date: str, leverage: int, units: float, start_cash: float):
    """
    Return the results of a backtest with the user specified parameters
    """
    return backtest(start_date = start_date, end_date = end_date, leverage = leverage, units = round(units), start_cash = start_cash)
