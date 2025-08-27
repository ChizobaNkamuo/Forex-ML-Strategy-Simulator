import "./App.css";
import '@mantine/core/styles.css';     // Base styles (required)
import '@mantine/dates/styles.css';   // Date picker styles
import { ChartComponent } from "./ChartComponent";
import { MantineProvider } from "@mantine/core";
import { DatePickerInput } from '@mantine/dates';

function StatText({name, value}) {
  return <li className="w-full text-white uppercase pb-3">
    <p className="text-xl">{value}</p>
    <p className="text-xl">{name}</p>
  </li>
}

function App() {
  const stats = [
    {name: "Win Rate", value: "67%"},
    {name: "Sharpe Ratio", value: "0.9"},
    {name: "ROI", value: "50%"},
    {name: "Profit Factor", value: "1.5"},
    {name: "Max Drawdown", value: "10%"},
  ];

  const equityData = [
      { time: "2018-12-22", value: 32.51 },
      { time: "2018-12-23", value: 31.11 },
      { time: "2018-12-24", value: 27.02 },
      { time: "2018-12-25", value: 27.32 },
      { time: "2018-12-26", value: 25.17 },
      { time: "2018-12-27", value: 28.89 },
      { time: "2018-12-28", value: 25.46 },
      { time: "2018-12-29", value: 23.92 },
      { time: "2018-12-30", value: 22.68 },
      { time: "2018-12-31", value: 22.67 },
  ];

  const startDate = new Date("2024-01-26");
  const endDate = new Date("2025-08-04");

  const equityCustomisation = {lineColor: "#2962FF", areaTopColor: "#2962FF", areaBottomColor: "oklch(27.9% 0.041 260.031)"}

  const candleStickData = [{ open: 10, high: 10.63, low: 9.49, close: 9.55, time: 1642427876 }, { open: 9.55, high: 10.30, low: 9.42, close: 9.94, time: 1642514276 }, { open: 9.94, high: 10.17, low: 9.92, close: 9.78, time: 1642600676 }, { open: 9.78, high: 10.59, low: 9.18, close: 9.51, time: 1642687076 }, { open: 9.51, high: 10.46, low: 9.10, close: 10.17, time: 1642773476 }, { open: 10.17, high: 10.96, low: 10.16, close: 10.47, time: 1642859876 }, { open: 10.47, high: 11.39, low: 10.40, close: 10.81, time: 1642946276 }, { open: 10.81, high: 11.60, low: 10.30, close: 10.75, time: 1643032676 }, { open: 10.75, high: 11.60, low: 10.49, close: 10.93, time: 1643119076 }, { open: 10.93, high: 11.53, low: 10.76, close: 10.96, time: 1643205476 }];

  const statItems = stats.map(stat => <StatText key = {stat.name} name = {stat.name} value = {stat.value}></StatText>);
  return (
    <div className="bg-slate-900 h-screen">
      <header className="px-5 pt-4  pb-2">
        <h1 className="text-white text-2xl font-bold">Forex Prediction Model Dashboard</h1>
      </header>

      <div className="grid grid-cols-4 gap-4 w-screen h-[calc(100vh-60px)] pt-1 px-5 pb-10">
        <div className="col-span-3 row-span-2 grid grid-rows-2 gap-4">
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-600">
              <ChartComponent name = "EQUITY CURVE" type="AreaSeries" data={equityData} customisation = {equityCustomisation}></ChartComponent>
          </div>
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-600">
              <ChartComponent name = "PRICE ACTION AND PREDICTIONS" type="CandlestickSeries" data={candleStickData} customisation = {[]}></ChartComponent>
          </div>

        </div>

        <div className="col-span-1 row-span-2 grid grid-rows-2 gap-4">
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-600">
            <h3 className="text-gray-400 text-sm tracking-wide mb-4 py m-0">PEFORMANCE METRICS</h3>
            <ul>{statItems}</ul>
          </div>
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-600">
            <h3 className="text-gray-400 text-sm tracking-wide mb-4 py m-0">TRADES MADE</h3>
              <MantineProvider defaultColorScheme="dark">
                <DatePickerInput
                  type="range"
                  placeholder="Select date range"
                  label="Analysis Period"
                  minDate={startDate}
                  maxDate={endDate}
                />
              </MantineProvider>

          </div>

        </div>

        
      </div>
    </div>
  )
}

export default App
