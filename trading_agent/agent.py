#!/usr/bin/env python3
"""AI-driven paper trading agent using Claude + Alpaca."""

import json
import os
from datetime import datetime, timedelta, timezone

from alpaca.data import NewsClient, StockHistoricalDataClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

load_dotenv()

WATCHLIST = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
MAX_POSITION_USD = 1000


class TradingAgent:
    def __init__(self):
        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self._init_llm()
        self.trading = TradingClient(key, secret, paper=True)
        self.market_data = StockHistoricalDataClient(key, secret)
        self.news_client = NewsClient(api_key=key, secret_key=secret)

    def _init_llm(self):
        """Configure the LLM backend from env vars.

        LLM_PROVIDER=anthropic (default) uses the Anthropic API.
        LLM_PROVIDER=local uses any OpenAI-compatible server, e.g.:
          - Ollama:    LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.1
          - LM Studio: LLM_BASE_URL=http://localhost:1234/v1   LLM_MODEL=<loaded model>
          - llama.cpp: LLM_BASE_URL=http://localhost:8080/v1
          - vLLM:      LLM_BASE_URL=http://localhost:8000/v1
        """
        self.provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        if self.provider == "anthropic":
            import anthropic
            self.llm = anthropic.Anthropic()
            self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        else:
            from openai import OpenAI
            base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
            api_key = os.environ.get("LLM_API_KEY", "not-needed")
            self.llm = OpenAI(base_url=base_url, api_key=api_key)
            self.model = os.environ.get("LLM_MODEL", "qwen2.5:3b")
        print(f"LLM backend: {self.provider} (model: {self.model})", flush=True)

    def _chat(self, prompt: str) -> str:
        """Send a single-turn prompt to the configured LLM and return text."""
        if self.provider == "anthropic":
            resp = self.llm.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        # OpenAI-compatible local server. response_format nudges JSON-capable
        # servers (Ollama, vLLM) to emit valid JSON; ignored by those that
        # don't support it.
        kwargs = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = self.llm.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except Exception:
            resp = self.llm.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def get_market_data(self, symbol: str) -> dict:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)

        bars = self.market_data.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=now)
        )
        quote = self.market_data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )

        closes = [bar.close for bar in bars[symbol]]
        latest_price = quote[symbol].ask_price or closes[-1]

        prices = closes[-20:]
        ma5 = round(sum(prices[-5:]) / 5, 2) if len(prices) >= 5 else None
        ma20 = round(sum(prices) / len(prices), 2) if prices else None
        change_pct = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) if len(closes) >= 2 else 0.0

        return {
            "symbol": symbol,
            "latest_price": round(float(latest_price), 2),
            "prices_10d": [round(p, 2) for p in prices[-10:]],
            "ma5": ma5,
            "ma20": ma20,
            "change_pct_1d": change_pct,
        }

    def get_news(self, symbol: str) -> list[str]:
        now = datetime.now(timezone.utc)
        try:
            resp = self.news_client.get_news(
                NewsRequest(symbols=[symbol], start=now - timedelta(days=3), end=now, limit=5)
            )
            return [article.headline for article in (resp.news if hasattr(resp, "news") else resp)]
        except Exception:
            return []

    def get_portfolio(self) -> dict:
        account = self.trading.get_account()
        positions = self.trading.get_all_positions()
        return {
            "cash": round(float(account.cash), 2),
            "portfolio_value": round(float(account.portfolio_value), 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_cost": round(float(p.avg_entry_price), 2),
                    "current_price": round(float(p.current_price), 2),
                    "market_value": round(float(p.market_value), 2),
                    "unrealized_pl": round(float(p.unrealized_pl), 2),
                }
                for p in positions
            ],
        }

    def ask_llm(self, symbol: str, market_data: dict, news: list[str], portfolio: dict) -> dict:
        position = next((p for p in portfolio["positions"] if p["symbol"] == symbol), None)
        position_str = (
            f"{position['qty']} shares @ avg ${position['avg_cost']} "
            f"(unrealized P/L: ${position['unrealized_pl']})"
            if position
            else "None"
        )

        prompt = f"""You are an AI stock trading analyst for a paper trading account (no real money at risk). Think only about the data given; do not invent facts.

## {symbol} Market Data
- Latest price: ${market_data['latest_price']}
- 1-day change: {market_data['change_pct_1d']}%
- Last 10 days closing prices: {market_data['prices_10d']}
- 5-day MA: ${market_data['ma5']} | 20-day MA: ${market_data['ma20']}

## Recent News (last 3 days)
{chr(10).join(f"- {h}" for h in news) if news else "- No recent news available"}

## Portfolio Context
- Cash available: ${portfolio['cash']:,.2f}
- Total portfolio value: ${portfolio['portfolio_value']:,.2f}
- Current {symbol} position: {position_str}

## Task
Analyze the price trend, moving averages, and news sentiment. Decide: BUY, SELL, or HOLD.
Rules: Max ${MAX_POSITION_USD} per BUY order. Only SELL if a position exists.

Respond ONLY with valid JSON (no markdown):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usd": <number if BUY, else null>,
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "<2-3 sentence explanation referencing the data>"
}}"""

        text = self._chat(prompt)
        start = text.find("{")
        end = text.rfind("}") + 1
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            # Small local models occasionally return malformed JSON; fail safe.
            return {
                "action": "HOLD",
                "amount_usd": None,
                "confidence": "LOW",
                "reasoning": f"Could not parse model response, defaulting to HOLD. Raw: {text[:120]}",
            }

    def execute_decision(self, symbol: str, decision: dict, market_data: dict, portfolio: dict) -> dict | None:
        action = decision["action"]

        if action == "BUY":
            amount = decision.get("amount_usd") or MAX_POSITION_USD
            amount = min(amount, portfolio["cash"], MAX_POSITION_USD)
            qty = int(amount / market_data["latest_price"])
            if qty < 1:
                return None
            order = self.trading.submit_order(
                MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            )
            return {"order_id": str(order.id), "symbol": symbol, "action": "BUY", "qty": qty}

        if action == "SELL":
            position = next((p for p in portfolio["positions"] if p["symbol"] == symbol), None)
            if not position:
                return None
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol, qty=position["qty"], side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                )
            )
            return {"order_id": str(order.id), "symbol": symbol, "action": "SELL", "qty": position["qty"]}

        return None

    def analyze(self, symbol: str) -> dict:
        portfolio = self.get_portfolio()
        market_data = self.get_market_data(symbol)
        news = self.get_news(symbol)
        decision = self.ask_llm(symbol, market_data, news, portfolio)

        order = None
        if decision["action"] != "HOLD":
            order = self.execute_decision(symbol, decision, market_data, portfolio)

        return {
            "symbol": symbol,
            "market_data": market_data,
            "news": news,
            "decision": decision,
            "order": order,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run(self, symbols: list[str] | None = None) -> list[dict]:
        symbols = symbols or WATCHLIST
        results = []
        for symbol in symbols:
            print(f"Analyzing {symbol}...", flush=True)
            result = self.analyze(symbol)
            d = result["decision"]
            print(f"  {d['action']} ({d['confidence']}) — {d['reasoning'][:80]}...")
            results.append(result)
        return results


if __name__ == "__main__":
    agent = TradingAgent()
    results = agent.run()
    print(json.dumps(results, indent=2, default=str))
