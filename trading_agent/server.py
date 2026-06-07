#!/usr/bin/env python3
"""Flask API server for the trading agent dashboard."""

import json
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from agent import TradingAgent, WATCHLIST

app = Flask(__name__)
CORS(app)

_agent: TradingAgent | None = None

def get_agent() -> TradingAgent:
    global _agent
    if _agent is None:
        _agent = TradingAgent()
    return _agent


@app.route("/api/portfolio")
def portfolio():
    try:
        data = get_agent().get_portfolio()
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/watchlist")
def watchlist():
    try:
        agent = get_agent()
        symbols = request.args.get("symbols", ",".join(WATCHLIST)).split(",")
        results = []
        for symbol in symbols:
            try:
                results.append(agent.get_market_data(symbol.strip()))
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})
        return jsonify({"ok": True, "data": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        body = request.get_json(silent=True) or {}
        symbols = body.get("symbols") or WATCHLIST
        if isinstance(symbols, str):
            symbols = [symbols]
        agent = get_agent()
        results = [agent.analyze(s) for s in symbols]
        return jsonify({"ok": True, "data": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/orders")
def orders():
    try:
        trading = get_agent().trading
        raw = trading.get_orders()
        data = [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value,
                "qty": float(o.qty or 0),
                "status": o.status.value,
                "filled_avg_price": float(o.filled_avg_price or 0),
                "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
            }
            for o in raw
        ]
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    print("Starting trading agent server on http://localhost:5000")
    print("Make sure your .env file has ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY")
    app.run(port=5000, debug=True)
