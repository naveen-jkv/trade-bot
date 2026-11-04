import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Shield,
  Zap,
  Play,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Layers,
  Database,
  Sliders,
  DollarSign,
  Clock,
  Briefcase,
  Terminal,
} from 'lucide-react';
import TradingViewChart from './components/TradingViewChart';

export default function App() {
  const [symbol, setSymbol] = useState('EUR/USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [marketData, setMarketData] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [history, setHistory] = useState({ forex_closed_trades: [], alpaca_orders: [] });
  const [memory, setMemory] = useState(null);
  const [latestSummary, setLatestSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeBottomTab, setActiveBottomTab] = useState('positions');

  // Order Desk state
  const [orderDeskType, setOrderDeskType] = useState('alpaca'); // 'alpaca' or 'forex'
  const [alpacaSymbol, setAlpacaSymbol] = useState('AAPL');
  const [orderSide, setOrderSide] = useState('buy');
  const [orderType, setOrderType] = useState('market');
  const [orderQty, setOrderQty] = useState(10);
  const [orderLimitPrice, setOrderLimitPrice] = useState('');
  const [orderStatusMsg, setOrderStatusMsg] = useState(null);

  // Fetch Market Data
  const fetchMarket = async (sym = symbol, tf = timeframe) => {
    try {
      const res = await fetch(`/api/market?symbol=${encodeURIComponent(sym)}&timeframe=${tf}`);
      if (res.ok) {
        const data = await res.json();
        setMarketData(data);
      }
    } catch (err) {
      console.error('Failed to fetch market data:', err);
    }
  };

  // Fetch Portfolio & Account
  const fetchPortfolio = async () => {
    try {
      const res = await fetch('/api/portfolio');
      if (res.ok) {
        const data = await res.json();
        setPortfolio(data);
      }
    } catch (err) {
      console.error('Failed to fetch portfolio:', err);
    }
  };

  // Fetch History
  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history?limit=30');
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  };

  // Fetch Memory & Diagnostics
  const fetchMemory = async () => {
    try {
      const res = await fetch('/api/memory');
      if (res.ok) {
        const data = await res.json();
        setMemory(data);
      }
    } catch (err) {
      console.error('Failed to fetch memory:', err);
    }
  };

  // Initial Load & Polling
  useEffect(() => {
    fetchMarket(symbol, timeframe);
    fetchPortfolio();
    fetchHistory();
    fetchMemory();

    const interval = setInterval(() => {
      fetchMarket(symbol, timeframe);
      fetchPortfolio();
    }, 10000);

    return () => clearInterval(interval);
  }, [symbol, timeframe]);

  // Execute Autonomous Cycle
  const handleRunCycle = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/cycle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, timeframe, force_synthetic: false }),
      });
      if (res.ok) {
        const data = await res.json();
        setLatestSummary(data);
        await Promise.all([fetchMarket(symbol, timeframe), fetchPortfolio(), fetchHistory(), fetchMemory()]);
      }
    } catch (err) {
      console.error('Cycle execution failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Close Forex Position
  const handleCloseForexPosition = async (ticketId) => {
    try {
      const res = await fetch('/api/orders/forex/close', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticket_id: ticketId }),
      });
      if (res.ok) {
        await Promise.all([fetchPortfolio(), fetchHistory()]);
      }
    } catch (err) {
      console.error('Close Forex failed:', err);
    }
  };

  // Submit Alpaca Simulated Order
  const handleSubmitAlpacaOrder = async () => {
    try {
      setOrderStatusMsg('Submitting simulated order...');
      const payload = {
        symbol: alpacaSymbol,
        qty: parseFloat(orderQty),
        side: orderSide,
        order_type: orderType,
        time_in_force: 'day',
        limit_price: orderType === 'limit' && orderLimitPrice ? parseFloat(orderLimitPrice) : null,
      };
      const res = await fetch('/api/orders/alpaca', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setOrderStatusMsg(`✅ ${data.status.toUpperCase()}: ${data.side.toUpperCase()} ${data.qty} ${data.symbol}`);
        await Promise.all([fetchPortfolio(), fetchHistory()]);
      } else {
        setOrderStatusMsg(`❌ Rejected: ${data.detail || data.rejection_reason || 'Order rejected'}`);
      }
    } catch (err) {
      setOrderStatusMsg(`❌ Error: ${err.message}`);
    }
  };

  // Close Alpaca Holding
  const handleCloseAlpacaPosition = async (sym) => {
    try {
      const res = await fetch('/api/orders/alpaca/close', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: sym }),
      });
      if (res.ok) {
        await Promise.all([fetchPortfolio(), fetchHistory()]);
      }
    } catch (err) {
      console.error('Close Alpaca failed:', err);
    }
  };

  const fxPort = portfolio?.forex || {
    balance: 10000,
    equity: 10000,
    free_margin: 10000,
    net_pnl: 0,
    return_pct: 0,
    win_rate_pct: 0,
    profit_factor: 1,
    max_drawdown_pct: 0,
    open_positions: [],
  };

  const alpAcct = portfolio?.alpaca || {
    portfolio_value: 100000,
    cash: 100000,
    buying_power: 200000,
    positions: [],
  };

  const currentAction = latestSummary?.proposal?.action || 'HOLD';
  const actionColor =
    currentAction === 'BUY' ? 'var(--color-buy)' : currentAction === 'SELL' ? 'var(--color-sell)' : 'var(--color-hold)';
  const actionBg =
    currentAction === 'BUY'
      ? 'var(--color-buy-bg)'
      : currentAction === 'SELL'
      ? 'var(--color-sell-bg)'
      : 'var(--color-hold-bg)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-canvas)' }}>
      {/* 1. TOP HEADER & TICKER BAR */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '10px 20px',
          background: 'var(--bg-surface)',
          borderBottom: '1px solid var(--border-subtle)',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '18px' }}>⚡</span>
            <span style={{ fontWeight: 700, fontSize: '15px', letterSpacing: '0.04em', color: '#f8fafc' }}>
              AUTONOMOUS QUANT TERMINAL
            </span>
          </div>

          {/* Instrument Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              style={{
                background: 'var(--bg-card)',
                color: '#f8fafc',
                border: '1px solid var(--border-subtle)',
                borderRadius: '6px',
                padding: '6px 12px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              <optgroup label="Forex Currency Pairs">
                <option value="EUR/USD">EUR/USD</option>
                <option value="GBP/USD">GBP/USD</option>
                <option value="USD/JPY">USD/JPY</option>
                <option value="AUD/USD">AUD/USD</option>
                <option value="USD/CAD">USD/CAD</option>
              </optgroup>
              <optgroup label="Alpaca Equities & Crypto">
                <option value="AAPL">AAPL (Apple)</option>
                <option value="MSFT">MSFT (Microsoft)</option>
                <option value="NVDA">NVDA (NVIDIA)</option>
                <option value="TSLA">TSLA (Tesla)</option>
                <option value="SPY">SPY (S&P 500)</option>
                <option value="QQQ">QQQ (Nasdaq 100)</option>
                <option value="BTC/USD">BTC/USD (Bitcoin)</option>
                <option value="ETH/USD">ETH/USD (Ethereum)</option>
              </optgroup>
            </select>

            {/* Timeframe Buttons */}
            <div style={{ display: 'flex', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '6px', overflow: 'hidden' }}>
              {['15m', '1h', '4h', '1d'].map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  style={{
                    padding: '5px 10px',
                    fontSize: '12px',
                    fontWeight: 600,
                    background: timeframe === tf ? 'var(--color-info-bg)' : 'transparent',
                    color: timeframe === tf ? 'var(--color-info)' : 'var(--text-muted)',
                  }}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          {/* Mark Price Ticker */}
          {marketData && (
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
              <span className="mono" style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc' }}>
                ${marketData.mark_price ? marketData.mark_price.toFixed(4) : '—'}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', background: 'var(--bg-card)', padding: '2px 6px', borderRadius: '4px' }}>
                {marketData.provenance}
              </span>
            </div>
          )}
        </div>

        {/* Right Side Status & Trigger */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ display: 'flex', gap: '6px', fontSize: '11px', fontWeight: 600 }}>
            <span style={{ background: 'rgba(34,197,94,0.15)', color: '#4ade80', padding: '3px 8px', borderRadius: '4px', border: '1px solid rgba(34,197,94,0.3)' }}>
              🟢 SWARM ACTIVE
            </span>
            <span style={{ background: 'rgba(59,130,246,0.15)', color: '#60a5fa', padding: '3px 8px', borderRadius: '4px', border: '1px solid rgba(59,130,246,0.3)' }}>
              🛡️ FIREWALL LOCKED
            </span>
            <span style={{ background: 'rgba(234,179,8,0.15)', color: '#facc15', padding: '3px 8px', borderRadius: '4px', border: '1px solid rgba(234,179,8,0.3)' }}>
              🟡 PAPER TRADING
            </span>
          </div>

          <button
            onClick={handleRunCycle}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: loading ? '#2563eb' : '#1d4ed8',
              color: '#ffffff',
              padding: '7px 16px',
              borderRadius: '6px',
              fontWeight: 700,
              fontSize: '13px',
              boxShadow: '0 2px 8px rgba(29, 78, 216, 0.4)',
            }}
          >
            <Play size={14} fill="currentColor" />
            {loading ? 'Analyzing...' : 'Run Cycle'}
          </button>
        </div>
      </header>

      {/* 2. KPI RIBBON */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '10px',
          padding: '12px 20px',
          background: 'var(--bg-canvas)',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div className="metric-box" style={{ background: 'var(--bg-surface)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Forex Virtual Equity
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', margin: '2px 0' }}>
            ${fxPort.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div style={{ fontSize: '11px', color: fxPort.net_pnl >= 0 ? 'var(--color-buy)' : 'var(--color-sell)', fontWeight: 600 }}>
            {fxPort.return_pct >= 0 ? '+' : ''}{fxPort.return_pct.toFixed(2)}% (${fxPort.net_pnl >= 0 ? '+' : ''}{fxPort.net_pnl.toFixed(2)})
          </div>
        </div>

        <div className="metric-box" style={{ background: 'var(--bg-surface)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Forex Cash & Free Margin
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', margin: '2px 0' }}>
            ${fxPort.balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            Free Margin: ${fxPort.free_margin.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        <div className="metric-box" style={{ background: 'var(--bg-surface)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Alpaca Simulated Cash
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', margin: '2px 0' }}>
            ${alpAcct.portfolio_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            Buying Power: ${alpAcct.buying_power.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        <div className="metric-box" style={{ background: 'var(--bg-surface)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Win Rate / Max DD
          </div>
          <div className="mono" style={{ fontSize: '18px', fontWeight: 700, color: '#f8fafc', margin: '2px 0' }}>
            {fxPort.win_rate_pct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            Profit Factor: {fxPort.profit_factor.toFixed(2)} | Max DD: {fxPort.max_drawdown_pct.toFixed(1)}%
          </div>
        </div>

        <div className="metric-box" style={{ background: 'var(--bg-surface)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Latest AI Signal
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '4px 0' }}>
            <span
              style={{
                background: actionBg,
                color: actionColor,
                fontWeight: 700,
                fontSize: '13px',
                padding: '2px 8px',
                borderRadius: '4px',
                border: `1px solid ${actionColor}`,
              }}
            >
              {currentAction}
            </span>
            <span className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              Conf: {latestSummary?.proposal?.confidence ? (latestSummary.proposal.confidence * 100).toFixed(0) : 65}%
            </span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            {symbol} | {marketData?.volatility_regime || 'NORMAL'}
          </div>
        </div>
      </div>

      {/* 3. MAIN SPLIT GRID (CHART & AI PANELS) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.8fr) minmax(320px, 1fr)', gap: '14px', padding: '14px 20px', flex: 1 }}>
        {/* LEFT: Financial Chart & Technicals */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '12px',
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: 600, color: '#f8fafc', fontSize: '13px' }}>
                {symbol} ({timeframe}) Interactive Candlestick Chart
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                Powered by TradingView Lightweight Charts
              </span>
            </div>

            <div style={{ flex: 1, minHeight: '420px' }}>
              <TradingViewChart bars={marketData?.bars || []} symbol={symbol} />
            </div>
          </div>

          {/* Technical Health Indicators Strip */}
          {marketData?.technical_state && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                gap: '8px',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '10px 14px',
              }}
            >
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>RSI (14)</div>
                <div className="mono" style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
                  {marketData.technical_state.rsi_14 ? marketData.technical_state.rsi_14.toFixed(1) : '50.0'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Trend</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#38bdf8' }}>
                  {marketData.technical_state.trend_direction ? marketData.technical_state.trend_direction.replace('_', ' ') : 'NEUTRAL'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Regime</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#f59e0b' }}>
                  {marketData.technical_state.volatility_regime || 'NORMAL'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>ATR (14)</div>
                <div className="mono" style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
                  {marketData.technical_state.atr_14 ? marketData.technical_state.atr_14.toFixed(5) : '0.0012'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>BB Width</div>
                <div className="mono" style={{ fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
                  {marketData.technical_state.bb_bandwidth ? marketData.technical_state.bb_bandwidth.toFixed(3) : '0.015'}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: AI Research Thesis & Quick Order Desk */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* AI Thesis Card */}
          <div
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '14px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <BrainIcon size={14} color="#38bdf8" /> AI Research Agent Thesis
              </span>
              <span
                style={{
                  background: actionBg,
                  color: actionColor,
                  fontWeight: 700,
                  fontSize: '11px',
                  padding: '2px 8px',
                  borderRadius: '4px',
                }}
              >
                {latestSummary?.hypothesis?.direction || 'HOLD'}
              </span>
            </div>

            <p style={{ color: 'var(--text-secondary)', fontSize: '12px', lineHeight: 1.5, marginBottom: '10px' }}>
              {latestSummary?.hypothesis?.hypothesis_text ||
                'Market in consolidation. Moving average envelopes aligned in compression zone. Awaiting breakout catalyst.'}
            </p>

            <div
              style={{
                background: 'rgba(239,68,68,0.1)',
                borderLeft: '3px solid #ef4444',
                padding: '6px 10px',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#fca5a5',
                marginBottom: '10px',
              }}
            >
              <strong>Invalidation:</strong>{' '}
              {latestSummary?.hypothesis?.invalidation_criteria || 'Close beyond opposite Bollinger Band.'}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
              <span>
                Entry: <strong style={{ color: '#f8fafc' }}>{latestSummary?.hypothesis?.proposed_entry?.toFixed(5) || '—'}</strong>
              </span>
              <span>
                SL: <strong style={{ color: '#ef4444' }}>{latestSummary?.hypothesis?.proposed_stop_loss?.toFixed(5) || '—'}</strong>
              </span>
              <span>
                TP: <strong style={{ color: '#22c55e' }}>{latestSummary?.hypothesis?.proposed_take_profit?.toFixed(5) || '—'}</strong>
              </span>
              <span>
                ML Consensus: <strong style={{ color: '#38bdf8' }}>{((latestSummary?.proposal?.ml_ensemble_prob_up || 0.65) * 100).toFixed(0)}%</strong>
              </span>
            </div>
          </div>

          {/* Quick Order Desk (Alpaca Mock Execution) */}
          <div
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '14px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontWeight: 700, color: '#f8fafc', fontSize: '13px' }}>
                📝 Simulated Order Desk
              </span>
              <span style={{ fontSize: '10px', color: '#facc15', background: 'rgba(234,179,8,0.15)', padding: '2px 6px', borderRadius: '4px' }}>
                Alpaca v2 Mock
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
              <div>
                <label style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Asset</label>
                <select
                  value={alpacaSymbol}
                  onChange={(e) => setAlpacaSymbol(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: '#f8fafc',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '4px',
                    padding: '5px 8px',
                    fontSize: '12px',
                  }}
                >
                  <option value="AAPL">AAPL (Stock)</option>
                  <option value="MSFT">MSFT (Stock)</option>
                  <option value="NVDA">NVDA (Stock)</option>
                  <option value="TSLA">TSLA (Stock)</option>
                  <option value="SPY">SPY (ETF)</option>
                  <option value="BTC/USD">BTC/USD (Crypto)</option>
                  <option value="ETH/USD">ETH/USD (Crypto)</option>
                </select>
              </div>

              <div>
                <label style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Order Type</label>
                <select
                  value={orderType}
                  onChange={(e) => setOrderType(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: '#f8fafc',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '4px',
                    padding: '5px 8px',
                    fontSize: '12px',
                  }}
                >
                  <option value="market">Market</option>
                  <option value="limit">Limit</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
              <div>
                <label style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Quantity</label>
                <input
                  type="number"
                  min="0.01"
                  step="1"
                  value={orderQty}
                  onChange={(e) => setOrderQty(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: '#f8fafc',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '4px',
                    padding: '5px 8px',
                    fontSize: '12px',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Side</label>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    onClick={() => setOrderSide('buy')}
                    style={{
                      flex: 1,
                      padding: '5px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 700,
                      background: orderSide === 'buy' ? 'var(--color-buy)' : 'var(--bg-card)',
                      color: orderSide === 'buy' ? '#ffffff' : 'var(--text-muted)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    BUY
                  </button>
                  <button
                    onClick={() => setOrderSide('sell')}
                    style={{
                      flex: 1,
                      padding: '5px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 700,
                      background: orderSide === 'sell' ? 'var(--color-sell)' : 'var(--bg-card)',
                      color: orderSide === 'sell' ? '#ffffff' : 'var(--text-muted)',
                      border: '1px solid var(--border-subtle)',
                    }}
                  >
                    SELL
                  </button>
                </div>
              </div>
            </div>

            {orderType === 'limit' && (
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Limit Price ($)</label>
                <input
                  type="number"
                  placeholder="e.g. 210.50"
                  value={orderLimitPrice}
                  onChange={(e) => setOrderLimitPrice(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'var(--bg-card)',
                    color: '#f8fafc',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '4px',
                    padding: '5px 8px',
                    fontSize: '12px',
                  }}
                />
              </div>
            )}

            <button
              onClick={handleSubmitAlpacaOrder}
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '6px',
                fontWeight: 700,
                fontSize: '12px',
                background: orderSide === 'buy' ? '#16a34a' : '#dc2626',
                color: '#ffffff',
              }}
            >
              Submit Simulated {orderSide.toUpperCase()} Order
            </button>

            {orderStatusMsg && (
              <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                {orderStatusMsg}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 4. BOTTOM DOCK (POSITIONS, HISTORY, RISK, MEMORY) */}
      <div
        style={{
          background: 'var(--bg-surface)',
          borderTop: '1px solid var(--border-subtle)',
          padding: '12px 20px',
        }}
      >
        {/* Dock Navigation */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px', marginBottom: '12px' }}>
          {[
            { id: 'positions', label: `💼 Open Positions (${fxPort.open_positions.length + alpAcct.positions.length})` },
            { id: 'history', label: `📜 Trade History (${history.forex_closed_trades.length})` },
            { id: 'risk', label: '🛡️ Risk Firewall Audit' },
            { id: 'memory', label: '🧠 AI Memory & Learning' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveBottomTab(tab.id)}
              style={{
                padding: '6px 14px',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 600,
                background: activeBottomTab === tab.id ? 'var(--color-info-bg)' : 'transparent',
                color: activeBottomTab === tab.id ? 'var(--color-info)' : 'var(--text-muted)',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 1: Positions */}
        {activeBottomTab === 'positions' && (
          <div>
            {fxPort.open_positions.length === 0 && alpAcct.positions.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', fontSize: '12px', padding: '10px 0' }}>
                No open positions. Run an autonomous cycle or submit a simulated order above.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
                      <th style={{ padding: '6px 8px' }}>Instrument</th>
                      <th style={{ padding: '6px 8px' }}>Engine</th>
                      <th style={{ padding: '6px 8px' }}>Action</th>
                      <th style={{ padding: '6px 8px' }}>Size</th>
                      <th style={{ padding: '6px 8px' }}>Entry Price</th>
                      <th style={{ padding: '6px 8px' }}>Mark Price</th>
                      <th style={{ padding: '6px 8px' }}>Unrealized P&L</th>
                      <th style={{ padding: '6px 8px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Forex Positions */}
                    {fxPort.open_positions.map((p) => (
                      <tr key={p.ticket_id} style={{ borderBottom: '1px solid rgba(35,53,77,0.4)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, color: '#f8fafc' }}>{p.symbol}</td>
                        <td style={{ padding: '6px 8px', color: 'var(--text-dim)' }}>Forex</td>
                        <td style={{ padding: '6px 8px', color: p.action === 'BUY' ? 'var(--color-buy)' : 'var(--color-sell)', fontWeight: 700 }}>
                          {p.action}
                        </td>
                        <td className="mono" style={{ padding: '6px 8px' }}>{p.lots} lots</td>
                        <td className="mono" style={{ padding: '6px 8px' }}>{p.entry_price.toFixed(5)}</td>
                        <td className="mono" style={{ padding: '6px 8px' }}>{p.current_price.toFixed(5)}</td>
                        <td className="mono" style={{ padding: '6px 8px', color: p.unrealized_pnl >= 0 ? 'var(--color-buy)' : 'var(--color-sell)', fontWeight: 600 }}>
                          ${p.unrealized_pnl >= 0 ? '+' : ''}{p.unrealized_pnl.toFixed(2)} ({p.unrealized_pips} pips)
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <button
                            onClick={() => handleCloseForexPosition(p.ticket_id)}
                            style={{
                              padding: '2px 8px',
                              borderRadius: '4px',
                              fontSize: '11px',
                              background: '#334155',
                              color: '#f8fafc',
                            }}
                          >
                            Close
                          </button>
                        </td>
                      </tr>
                    ))}

                    {/* Alpaca Positions */}
                    {alpAcct.positions.map((ap) => (
                      <tr key={ap.symbol} style={{ borderBottom: '1px solid rgba(35,53,77,0.4)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, color: '#f8fafc' }}>{ap.symbol}</td>
                        <td style={{ padding: '6px 8px', color: '#facc15' }}>Alpaca Mock</td>
                        <td style={{ padding: '6px 8px', color: 'var(--color-buy)', fontWeight: 700 }}>LONG</td>
                        <td className="mono" style={{ padding: '6px 8px' }}>{ap.qty} shares</td>
                        <td className="mono" style={{ padding: '6px 8px' }}>${ap.avg_entry_price.toFixed(2)}</td>
                        <td className="mono" style={{ padding: '6px 8px' }}>${ap.current_price.toFixed(2)}</td>
                        <td className="mono" style={{ padding: '6px 8px', color: ap.unrealized_pl >= 0 ? 'var(--color-buy)' : 'var(--color-sell)', fontWeight: 600 }}>
                          ${ap.unrealized_pl >= 0 ? '+' : ''}{ap.unrealized_pl.toFixed(2)} ({(ap.unrealized_plpc * 100).toFixed(2)}%)
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <button
                            onClick={() => handleCloseAlpacaPosition(ap.symbol)}
                            style={{
                              padding: '2px 8px',
                              borderRadius: '4px',
                              fontSize: '11px',
                              background: '#334155',
                              color: '#f8fafc',
                            }}
                          >
                            Liquidate
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: History */}
        {activeBottomTab === 'history' && (
          <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
                  <th style={{ padding: '4px 8px' }}>Ticket</th>
                  <th style={{ padding: '4px 8px' }}>Symbol</th>
                  <th style={{ padding: '4px 8px' }}>Action</th>
                  <th style={{ padding: '4px 8px' }}>Volume</th>
                  <th style={{ padding: '4px 8px' }}>Entry</th>
                  <th style={{ padding: '4px 8px' }}>Exit</th>
                  <th style={{ padding: '4px 8px' }}>Realized P&L</th>
                  <th style={{ padding: '4px 8px' }}>Reason</th>
                </tr>
              </thead>
              <tbody>
                {history.forex_closed_trades.map((t) => (
                  <tr key={t.ticket_id} style={{ borderBottom: '1px solid rgba(35,53,77,0.3)' }}>
                    <td style={{ padding: '4px 8px', color: 'var(--text-dim)' }}>{t.ticket_id}</td>
                    <td style={{ padding: '4px 8px', fontWeight: 600, color: '#f8fafc' }}>{t.symbol}</td>
                    <td style={{ padding: '4px 8px', color: t.action === 'BUY' ? 'var(--color-buy)' : 'var(--color-sell)' }}>{t.action}</td>
                    <td className="mono" style={{ padding: '4px 8px' }}>{t.volume_lots}L</td>
                    <td className="mono" style={{ padding: '4px 8px' }}>{t.entry_price.toFixed(5)}</td>
                    <td className="mono" style={{ padding: '4px 8px' }}>{t.exit_price.toFixed(5)}</td>
                    <td className="mono" style={{ padding: '4px 8px', color: t.pnl_dollars >= 0 ? 'var(--color-buy)' : 'var(--color-sell)', fontWeight: 600 }}>
                      ${t.pnl_dollars >= 0 ? '+' : ''}{t.pnl_dollars.toFixed(2)}
                    </td>
                    <td style={{ padding: '4px 8px', color: 'var(--text-dim)' }}>{t.exit_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Risk Firewall */}
        {activeBottomTab === 'risk' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px', fontSize: '12px' }}>
            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '4px' }}>🛡️ Hard Safety Boundaries</div>
              <ul style={{ listStyle: 'none', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                <li>Max Risk / Trade: <strong>1.5%</strong> (Hard Cap: 2.5%)</li>
                <li>Daily Drawdown Cutoff: <strong>3.5%</strong></li>
                <li>Max Total Drawdown: <strong>8.0%</strong></li>
                <li>Min Risk-to-Reward: <strong>1:1.50</strong></li>
              </ul>
            </div>
            <div style={{ background: 'var(--bg-card)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: '#22c55e', marginBottom: '4px' }}>✅ Firewall Compliance Status</div>
              <div style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                All orders are pre-screened before execution. Orders violating RR or portfolio caps are blocked automatically.
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Memory */}
        {activeBottomTab === 'memory' && (
          <div style={{ maxHeight: '180px', overflowY: 'auto' }}>
            {memory?.lessons && memory.lessons.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
                    <th style={{ padding: '4px 8px' }}>Instrument</th>
                    <th style={{ padding: '4px 8px' }}>Regime</th>
                    <th style={{ padding: '4px 8px' }}>Mistake Analysis</th>
                    <th style={{ padding: '4px 8px' }}>Rule Learned</th>
                    <th style={{ padding: '4px 8px' }}>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {memory.lessons.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(35,53,77,0.3)' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 600, color: '#f8fafc' }}>{l.symbol}</td>
                      <td style={{ padding: '4px 8px', color: '#f59e0b' }}>{l.regime}</td>
                      <td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>{l.mistake_analysis}</td>
                      <td style={{ padding: '4px 8px', color: '#38bdf8' }}>{l.rule_learned}</td>
                      <td className="mono" style={{ padding: '4px 8px', color: l.outcome_pnl >= 0 ? 'var(--color-buy)' : 'var(--color-sell)' }}>
                        ${l.outcome_pnl ? l.outcome_pnl.toFixed(2) : '0.00'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ color: 'var(--text-dim)', fontSize: '12px', padding: '10px 0' }}>
                No lessons recorded yet. Completed trades automatically generate post-mortems in SQLite.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function BrainIcon({ size = 14, color = '#38bdf8' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z" />
    </svg>
  );
}
