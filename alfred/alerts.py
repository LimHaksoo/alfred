"""Actions: voice prompt, path light indicator, family notification (Telegram).

Secrets come from the environment (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID);
never commit them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import requests

VOICE = {
    "voice_checkin": "Are you okay? If you need help, just say help.",
    "bed_exit": "Good evening. Please sit on the bed for ten seconds before you stand up. I turned on the path light.",
    "sway": "Please hold on to the wall for a moment.",
}


def say(text: str):
    """Speak without blocking the video loop (macOS `say`, else print)."""
    if shutil.which("say"):
        subprocess.Popen(["say", text])
    print(f"[voice] {text}")


class Notifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat = os.environ.get("TELEGRAM_CHAT_ID")
        self.log: list[dict] = []

    def family(self, text: str, detected_at: float | None = None) -> dict:
        """Send and record latency from detection to Telegram accepting the message (number A)."""
        t_send = time.time()
        rec = {"text": text, "sent_at": t_send, "ok": False}
        if self.token and self.chat:
            try:
                r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                  json={"chat_id": self.chat, "text": text}, timeout=10)
                rec["ok"] = r.ok
            except requests.RequestException as e:
                rec["error"] = str(e)
        else:
            rec["error"] = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
        rec["delivered_at"] = time.time()
        if detected_at is not None:
            rec["latency_s"] = round(rec["delivered_at"] - detected_at, 3)
        self.log.append(rec)
        print(f"[family] {text} -> {rec.get('latency_s')}s ok={rec['ok']}")
        return rec
