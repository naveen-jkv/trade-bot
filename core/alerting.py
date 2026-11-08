"""
Operational Alerting & Notification Dispatcher.
Provides multi-channel notifications (Telegram Bot, Discord Webhook, Generic HTTP Webhooks)
for critical trading events: Kill Switch activations, daily drawdown ceiling breaches,
emergency position flattens, and trade execution updates.
Designed to fail safely: network errors or missing tokens NEVER crash the trading engine.
"""

from enum import Enum
import logging
from typing import Dict, Optional, Any
import requests

from config import DEFAULT_CONFIG, AlertSettings

logger = logging.getLogger("AlertManager")


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertEvent(str, Enum):
    KILL_SWITCH_ENGAGED = "KILL_SWITCH_ENGAGED"
    KILL_SWITCH_DISENGAGED = "KILL_SWITCH_DISENGAGED"
    DAILY_DRAWDOWN_BREACH = "DAILY_DRAWDOWN_BREACH"
    TOTAL_DRAWDOWN_BREACH = "TOTAL_DRAWDOWN_BREACH"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    ORDER_REJECTED = "ORDER_REJECTED"
    SYSTEM_STARTUP = "SYSTEM_STARTUP"


SEVERITY_ICONS = {
    AlertSeverity.INFO: "🟢",
    AlertSeverity.WARNING: "🟡",
    AlertSeverity.CRITICAL: "🔴",
}


class AlertManager:
    """Dispatches operational alerts to configured endpoints with non-blocking resilience."""

    def __init__(self, settings: Optional[AlertSettings] = None, timeout: float = 4.0):
        self.settings = settings or DEFAULT_CONFIG.alerts
        self.timeout = timeout

    def send_alert(
        self,
        event: AlertEvent,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Dispatches alert across all active communication channels.
        Returns True if at least one remote channel dispatched successfully (or if alerts are disabled).
        """
        icon = SEVERITY_ICONS.get(severity, "⚪")
        formatted_title = f"{icon} [{severity.value}] {title}"
        full_text = f"{formatted_title}\n\nEvent: {event.value}\n{message}"

        if metadata:
            meta_str = "\n".join(f"• {k}: {v}" for k, v in metadata.items())
            full_text += f"\n\nDetails:\n{meta_str}"

        # Always log locally first
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"ALERT: {title} - {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"ALERT: {title} - {message}")
        else:
            logger.info(f"ALERT: {title} - {message}")

        if not self.settings.enabled:
            return True

        dispatched = False

        # 1. Telegram Dispatch
        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            try:
                tg_url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
                payload = {
                    "chat_id": self.settings.telegram_chat_id,
                    "text": full_text,
                    "parse_mode": "Markdown",
                }
                resp = requests.post(tg_url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    dispatched = True
                else:
                    logger.debug(f"Telegram alert delivery returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Failed to dispatch Telegram alert: {e}")

        # 2. Discord Webhook Dispatch
        if self.settings.discord_webhook_url:
            try:
                color_map = {
                    AlertSeverity.INFO: 0x2ECC71,      # Green
                    AlertSeverity.WARNING: 0xF1C40F,   # Yellow
                    AlertSeverity.CRITICAL: 0xE74C3C,  # Red
                }
                embed = {
                    "title": formatted_title,
                    "description": message,
                    "color": color_map.get(severity, 0x95A5A6),
                    "fields": [
                        {"name": "Event", "value": event.value, "inline": True},
                        {"name": "Severity", "value": severity.value, "inline": True},
                    ],
                }
                if metadata:
                    for k, v in list(metadata.items())[:5]:
                        embed["fields"].append({"name": str(k), "value": str(v), "inline": True})

                payload = {"embeds": [embed]}
                resp = requests.post(self.settings.discord_webhook_url, json=payload, timeout=self.timeout)
                if resp.status_code in (200, 204):
                    dispatched = True
                else:
                    logger.debug(f"Discord webhook returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Failed to dispatch Discord webhook: {e}")

        # 3. Generic JSON Webhook Dispatch
        if self.settings.generic_webhook_url:
            try:
                payload = {
                    "event": event.value,
                    "severity": severity.value,
                    "title": title,
                    "message": message,
                    "metadata": metadata or {},
                }
                resp = requests.post(self.settings.generic_webhook_url, json=payload, timeout=self.timeout)
                if resp.status_code in (200, 201, 202, 204):
                    dispatched = True
                else:
                    logger.debug(f"Generic webhook returned HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"Failed to dispatch generic webhook: {e}")

        return dispatched


# Shared default AlertManager singleton
default_alert_manager = AlertManager()
