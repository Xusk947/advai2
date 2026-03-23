import requests
import os
from typing import Optional
from src.config import BOT_TOKEN, CHAT_ID

def send_telegram_document(file_path: str, caption: Optional[str] = None) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram BOT_TOKEN or CHAT_ID not set. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
            
        if response.status_code != 200:
            print(f"Failed to send Telegram document: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram document: {e}")

def send_telegram_video(file_path: str, caption: Optional[str] = None) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    try:
        with open(file_path, "rb") as f:
            files = {"video": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
            
        if response.status_code != 200:
            print(f"Failed to send Telegram video: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram video: {e}")
