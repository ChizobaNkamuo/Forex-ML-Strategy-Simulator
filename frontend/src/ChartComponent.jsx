
import { AreaSeries, CandlestickSeries,createChart, ColorType, createSeriesMarkers } from "lightweight-charts";
import React, { useEffect, useRef } from "react";

const chartTypes = {
    CandlestickSeries: CandlestickSeries,
    AreaSeries: AreaSeries
}

export const ChartComponent = props => {
    const backgroundColor = "oklch(27.9% 0.041 260.031)";
    const textColor = "white";
    const {
        name,
        type,
        data,
        customisation,
        markers,
        CustomLegend
    } = props;

    const chartContainerRef = useRef();

    useEffect(
        () => {
            const handleResize = () => {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth, height: chartContainerRef.current.clientHeight });
            };

            const chart = createChart(chartContainerRef.current, {
                layout: {
                    background: { type: ColorType.Solid, color: backgroundColor },
                    textColor,
                },
                width: chartContainerRef.current.clientWidth,
                height: chartContainerRef.current.clientHeight,
            });
            chart.applyOptions({
                rightPriceScale: {
                    scaleMargins: {
                        top: 0.3, // leave some space for the legend
                        bottom: 0,
                    },
                },
                grid: {
                    vertLines: {
                        visible: false,
                    },
                    horzLines: {
                        visible: false,
                    },
                },
            })
            chart.timeScale().fitContent();

            const newSeries = chart.addSeries(chartTypes[type], {customisation});
            newSeries.setData(data);

            if (markers) {
                createSeriesMarkers(newSeries, markers);
            }

            window.addEventListener("resize", handleResize);

            return () => {
                window.removeEventListener("resize", handleResize);

                chart.remove();
            };
        },
        [data]
    );

    return (
    <div className="h-full w-full relative" ref={chartContainerRef} >
        <div className="absolute left-20 top-3 z-10 font-light leading-[18px] text-gray-400 text-xl">
            {name}
            {CustomLegend && <CustomLegend />}
        </div>
    </div>
    );
};