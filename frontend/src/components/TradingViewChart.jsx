import React, { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries, LineSeries, HistogramSeries, ColorType } from 'lightweight-charts';

export default function TradingViewChart({ bars = [], activePositions = [], symbol = "EUR/USD" }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const ema9SeriesRef = useRef(null);
  const ema21SeriesRef = useRef(null);
  const ema50SeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight || 480,
      layout: {
        background: { type: ColorType.Solid, color: '#090e17' },
        textColor: '#94a3b8',
        fontSize: 12,
        fontFamily: "'Inter', sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(35, 53, 77, 0.45)' },
        horzLines: { color: 'rgba(35, 53, 77, 0.45)' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#23354d',
        fixLeftEdge: true,
      },
      rightPriceScale: {
        borderColor: '#23354d',
        scaleMargins: {
          top: 0.1,
          bottom: 0.22,
        },
      },
      crosshair: {
        mode: 0,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    const ema9 = chart.addSeries(LineSeries, {
      color: '#38bdf8',
      lineWidth: 1.5,
      title: 'EMA 9',
    });

    const ema21 = chart.addSeries(LineSeries, {
      color: '#a855f7',
      lineWidth: 1.5,
      title: 'EMA 21',
    });

    const ema50 = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 1.5,
      title: 'EMA 50',
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: '#3b82f6',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // set as an overlay by setting a blank priceScaleId
    });

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    ema9SeriesRef.current = ema9;
    ema21SeriesRef.current = ema21;
    ema50SeriesRef.current = ema50;
    volumeSeriesRef.current = volumeSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight || 480,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  // Update Data
  useEffect(() => {
    if (!candleSeriesRef.current || !bars || bars.length === 0) return;

    try {
      // Sort bars ascending by time
      const sortedBars = [...bars].sort((a, b) => a.time - b.time);

      const candleData = sortedBars.map((b) => ({
        time: b.time,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }));
      candleSeriesRef.current.setData(candleData);

      const ema9Data = sortedBars
        .filter((b) => b.ema_9 !== undefined && !isNaN(b.ema_9))
        .map((b) => ({ time: b.time, value: b.ema_9 }));
      if (ema9SeriesRef.current) ema9SeriesRef.current.setData(ema9Data);

      const ema21Data = sortedBars
        .filter((b) => b.ema_21 !== undefined && !isNaN(b.ema_21))
        .map((b) => ({ time: b.time, value: b.ema_21 }));
      if (ema21SeriesRef.current) ema21SeriesRef.current.setData(ema21Data);

      const ema50Data = sortedBars
        .filter((b) => b.ema_50 !== undefined && !isNaN(b.ema_50))
        .map((b) => ({ time: b.time, value: b.ema_50 }));
      if (ema50SeriesRef.current) ema50SeriesRef.current.setData(ema50Data);

      const volData = sortedBars.map((b) => ({
        time: b.time,
        value: b.volume || 1000,
        color: b.close >= b.open ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      }));
      if (volumeSeriesRef.current) volumeSeriesRef.current.setData(volData);

      chartRef.current?.timeScale().fitContent();
    } catch (err) {
      console.error("Error setting chart data:", err);
    }
  }, [bars]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '440px' }}>
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%', minHeight: '440px' }} />
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '14px',
          display: 'flex',
          gap: '12px',
          fontSize: '11px',
          fontWeight: 600,
          background: 'rgba(9, 14, 23, 0.75)',
          padding: '4px 10px',
          borderRadius: '4px',
          border: '1px solid #23354d',
          zIndex: 5,
        }}
      >
        <span style={{ color: '#f8fafc' }}>{symbol}</span>
        <span style={{ color: '#38bdf8' }}>EMA 9</span>
        <span style={{ color: '#a855f7' }}>EMA 21</span>
        <span style={{ color: '#f59e0b' }}>EMA 50</span>
      </div>
    </div>
  );
}

