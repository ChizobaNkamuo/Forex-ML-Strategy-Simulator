import "./App.css";
import "@mantine/core/styles.css";
import "@mantine/dates/styles.css";
import axios from "axios";
import { ChartComponent } from "./ChartComponent";
import { MantineProvider } from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import { useState, useEffect } from "react";
import spinnerImg from "./assets/Dual_Ring.svg"

function StatText({name, value}) {
  return <li className="w-full text-white uppercase pb-3">
    <p className="text-xl">{value}</p>
    <p className="text-xl">{name}</p>
  </li>
}

function Spinner() {
  return <img src={spinnerImg} className="px-"></img>
}

function CandleStickLegend() {
  return <div>
      <div className="text-green-500 text-sm font-bold">▲ Buy Signal</div>
      <div className="text-red-500 text-sm font-bold">▼ Sell Signal</div>
  </div>
}

function App() {
  const oldstats = [
    {name: "Win Rate", value: "67%"},
    {name: "Sharpe Ratio", value: "0.9"},
    {name: "ROI", value: "50%"},
    {name: "Profit Factor", value: "1.5"},
    {name: "Max Drawdown", value: "10%"},
  ];
  const startDate = "2024-04-25";
  const endDate = "2025-08-04";
  const [equityData, setEquityData] = useState([]);
  const [candleStickData, setCandleStickData] = useState([]);
  const [markers, setMarkers] = useState([]);
  const [stats, setStats] = useState({});
  const [trades, setTrades] = useState([]);
  const [date, setDate] = useState([startDate, endDate]);
  const [leverage, setLeverage] = useState(10);
  const [leverageError, setLeverageError] = useState(false);
  const [lotSize, setLotSize] = useState(8);
  const [lotError, setLotError] = useState(false);
  const [cash, setCash] = useState(100000.0);
  const [cashError, setCashError] = useState(false);
  const [loading, setLoading] = useState(false);

  const equityCustomisation = {lineColor: "#2962FF", areaTopColor: "#2962FF", areaBottomColor: "oklch(27.9% 0.041 260.031)"}
  function runModel() {
      setLoading(true);
      axios.get("http://127.0.0.1:8000/backtest",{
        params: {
          start_date: date[0],
          end_date: date[1],
          leverage: leverage,
          units: lotSize * 100000,
          start_cash: cash
        }
      })
      .then((res) => {
        setEquityData(res.data["equity_curve"]);
        setCandleStickData(res.data["candle_sticks"]);
        setMarkers(res.data["markers"]);
        setStats(res.data["stats"]);
        setTrades(res.data["trades"]);
        setLoading(false)
      })
      .catch((err) => console.error("Error fetching data:", err))
  };

  function checkInt(setValue, setError, value) {
    const intValue = parseInt(value)
    if (!isNaN(intValue) && isFinite(value) && intValue > 0) {
      setValue(intValue);
      setError(false);
    }else{
      setError(true);
    }
  }
  function checkFloat(setValue, setError, value) {
    const floatValue = parseFloat(value)
    if (!isNaN(floatValue) && isFinite(value) && floatValue > 0) {
      setValue(floatValue);
      setError(false);
    }else{
      setError(true);
    }
  }

  useEffect(() => runModel(), []);

  return (
    <div className="bg-slate-900 h-screen">
      <header className="px-5 pt-4  pb-2">
        <h1 className="text-white text-2xl font-bold">Forex Prediction Model Dashboard</h1>
      </header>

      <div className="grid grid-cols-6 gap-4 w-screen h-[calc(100vh-60px)] pt-1 px-5 pb-10">
        <div className="col-span-3 row-span-3 grid grid-rows-2 gap-4">
          <div className="grid place-items-center bg-slate-800 rounded-xl p-4 border border-slate-600">
              {loading ? Spinner(): 
              <ChartComponent name = "EQUITY CURVE" type="AreaSeries" data={equityData} customisation = {equityCustomisation}></ChartComponent>}
          </div>
          <div className="grid place-items-center bg-slate-800 rounded-xl p-4 border border-slate-600">
            {loading ? Spinner(): 
            <ChartComponent name = "PRICE ACTION AND PREDICTIONS" type="CandlestickSeries" data={candleStickData} customisation = {[]} markers = {markers} CustomLegend = {CandleStickLegend}></ChartComponent>}
          </div>
        </div>

        <div className={`${loading ? "grid place-items-center" : ""} "col-span-1 row-span-3 bg-slate-800 rounded-xl p-4 border border-slate-600`}>
          {loading ? Spinner(): 
          <>
          <h3 className="text-gray-400 text-sm tracking-wide mb-4 py m-0">PEFORMANCE METRICS</h3>
          <ul>{Object.keys(stats).map((key) => (<StatText key = {key} name = {key} value = {stats[key]}></StatText>))}</ul>
          </>}
        </div>

        <div className={`${loading ? "grid place-items-center" : ""} scrollbar col-span-1 row-span-3 bg-slate-800 rounded-xl p-4 border border-slate-600 overflow-y-scroll`}>
          {loading ? Spinner(): 
          <>
          <h3 className="text-gray-400 text-sm tracking-wide mb-4 py m-0">TRADES MADE</h3>
          <ul>{trades.map((key) => (<li className="border-b text-sm border-slate-600" key = {key}>{key}</li>))}</ul>
          </>}
        </div>

        <div className="col-span-1 row-span-3 bg-slate-800 rounded-xl p-4 border border-slate-600">
            <h3 className="text-gray-400 text-sm tracking-wide mb-4 py m-0">SETTINGS</h3>
              <MantineProvider defaultColorScheme="dark">
                <DatePickerInput
                  type="range"
                  label="Analysis Period"
                  minDate={startDate}
                  maxDate={endDate}
                  value={date}
                  onChange={setDate}
                />
              </MantineProvider>
              <p className="font-medium text-sm pt-6">Leverage</p>
              <input defaultValue={leverage} className="rounded-lg p-1 bg-[#2e2e2e] border border-[#424242]" onChange={e => checkInt(setLeverage, setLeverageError, e.target.value)} ></input>
              <p className={`${leverageError ? "visible": "invisible"} text-red-500 font-medium text-sm`}>Invalid Leverage</p>

              <p className="font-medium text-sm pt-3">Lot Size</p>
              <input defaultValue={lotSize} className="rounded-lg p-1 bg-[#2e2e2e] border border-[#424242]" onChange={e => checkFloat(setLotSize, setLotError, e.target.value)} ></input>
              <p className={`${lotError ? "visible": "invisible"} text-red-500 font-medium text-sm`}>Invalid Lot Size</p>

              <p className="font-medium text-sm pt-3">Start Cash ($)</p>
              <input defaultValue={cash} className="rounded-lg p-1 bg-[#2e2e2e] border border-[#424242]" onChange={e => checkFloat(setCash, setCashError, e.target.value)} ></input>
              <p className={`${cashError ? "visible": "invisible"} text-red-500 font-medium text-sm`}>Invalid Cash Value</p>

              <button className="disabled:opacity-50 disabled:hover:border-slate-900 bg-slate-900 p-2 my-5" disabled={cashError || lotError || leverageError} onClick={runModel}>Run Model</button>
        </div>
        
      </div>
    </div>
  )
}

export default App
