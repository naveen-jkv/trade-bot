"""
Unit tests for Multi-Channel Operational Alerting (Telegram, Discord, Webhooks).
"""

from unittest.mock import patch, MagicMock
import requests
import pytest

from core.alerting import AlertManager, AlertEvent, AlertSeverity
from config import AlertSettings


def test_alert_manager_disabled():
    """Verify that disabled alerting returns cleanly without invoking network requests."""
    settings = AlertSettings(enabled=False)
    am = AlertManager(settings=settings)

    with patch("requests.post") as mock_post:
        res = am.send_alert(
            event=AlertEvent.TRADE_OPENED,
            severity=AlertSeverity.INFO,
            title="Trade Opened",
            message="Opened BUY 1.0 lots on EUR/USD.",
        )
        assert res is True
        mock_post.assert_not_called()


def test_alert_manager_telegram_dispatch():
    """Verify Telegram Bot API payload formatting and successful dispatch."""
    settings = AlertSettings(
        enabled=True,
        telegram_bot_token="TEST_BOT_TOKEN_123",
        telegram_chat_id="CHAT_999",
    )
    am = AlertManager(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = am.send_alert(
            event=AlertEvent.KILL_SWITCH_ENGAGED,
            severity=AlertSeverity.CRITICAL,
            title="Kill Switch Activated",
            message="Emergency kill switch engaged by operator.",
            metadata={"source": "api_endpoint", "operator": "admin"},
        )
        assert res is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "https://api.telegram.org/botTEST_BOT_TOKEN_123/sendMessage" in args[0]
        payload = kwargs["json"]
        assert payload["chat_id"] == "CHAT_999"
        assert "CRITICAL" in payload["text"]
        assert "KILL_SWITCH_ENGAGED" in payload["text"]


def test_alert_manager_discord_webhook_dispatch():
    """Verify Discord Webhook embed payload construction and dispatch."""
    settings = AlertSettings(
        enabled=True,
        discord_webhook_url="https://discord.com/api/webhooks/123/abc",
    )
    am = AlertManager(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = am.send_alert(
            event=AlertEvent.DAILY_DRAWDOWN_BREACH,
            severity=AlertSeverity.CRITICAL,
            title="Daily Drawdown Ceiling Reached",
            message="Daily loss exceeded 3.5% threshold.",
            metadata={"daily_loss_pct": 3.8, "baseline_equity": 15000.0},
        )
        assert res is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://discord.com/api/webhooks/123/abc"
        payload = kwargs["json"]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert "CRITICAL" in embed["title"]
        assert embed["color"] == 0xE74C3C  # Red for CRITICAL


def test_alert_manager_network_error_resilience():
    """Verify that network exceptions (timeouts, DNS failures) are handled gracefully without crashing."""
    settings = AlertSettings(
        enabled=True,
        telegram_bot_token="TEST_BOT_TOKEN_123",
        telegram_chat_id="CHAT_999",
    )
    am = AlertManager(settings=settings)

    with patch("requests.post", side_effect=requests.exceptions.ConnectTimeout("Network timed out")):
        # Must not raise exception
        res = am.send_alert(
            event=AlertEvent.EMERGENCY_FLATTEN,
            severity=AlertSeverity.CRITICAL,
            title="Emergency Flatten Triggered",
            message="Liquidated all active positions.",
        )
        assert res is False  # Delivery failed, but no crash
