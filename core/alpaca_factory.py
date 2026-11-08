"""
Alpaca Client Factory.
Instantiates the appropriate Alpaca client based on system configuration and environment variables:
- 'LIVE': LiveAlpacaClient connected to real money endpoint (https://api.alpaca.markets)
- 'PAPER': LiveAlpacaClient connected to Alpaca paper endpoint (https://paper-api.alpaca.markets), or fallback to MockAlpacaClient if keys are not set
- 'MOCK': MockAlpacaClient (local offline simulated execution engine)
"""

import logging
import os
from typing import Optional

from config import DEFAULT_CONFIG, SystemConfig
from core.alpaca_interface import AlpacaClientInterface
from core.alpaca_live_client import LiveAlpacaClient
from core.alpaca_mock_client import MockAlpacaClient
from database.memory import TradingMemoryDB

logger = logging.getLogger("AlpacaFactory")


def create_alpaca_client(
    config: Optional[SystemConfig] = None,
    db: Optional[TradingMemoryDB] = None,
) -> AlpacaClientInterface:
    """
    Factory function to create either LiveAlpacaClient or MockAlpacaClient.
    """
    cfg = config or DEFAULT_CONFIG
    alpaca_cfg = cfg.alpaca
    mode = getattr(alpaca_cfg, "trading_mode", os.getenv("TRADING_MODE", "PAPER")).upper()
    api_key = (getattr(alpaca_cfg, "api_key", None) or os.getenv("ALPACA_API_KEY", "")).strip()
    secret_key = (getattr(alpaca_cfg, "secret_key", None) or os.getenv("ALPACA_SECRET_KEY", "")).strip()

    has_credentials = bool(api_key and secret_key)

    if mode == "LIVE":
        if not has_credentials:
            logger.warning(
                "TRADING_MODE is set to 'LIVE' (Real Money), but ALPACA_API_KEY / ALPACA_SECRET_KEY "
                "are missing. Falling back to MockAlpacaClient for safety."
            )
            return MockAlpacaClient(config=cfg, db=db)
        logger.info("Initializing LiveAlpacaClient in LIVE REAL-MONEY mode.")
        return LiveAlpacaClient(config=cfg, db=db, is_paper=False)

    elif mode == "PAPER":
        if has_credentials:
            logger.info("Initializing LiveAlpacaClient in ALPACA PAPER mode (remote paper trading).")
            return LiveAlpacaClient(config=cfg, db=db, is_paper=True)
        else:
            logger.info("No Alpaca credentials supplied. Initializing offline MockAlpacaClient.")
            return MockAlpacaClient(config=cfg, db=db)

    else:  # 'MOCK'
        logger.info("Initializing offline MockAlpacaClient.")
        return MockAlpacaClient(config=cfg, db=db)

