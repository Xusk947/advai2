import requests
from typing import Optional
from src.config import BOT_TOKEN, CHAT_ID

def send_telegram_message(message: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def send_telegram_document(file_path: str, caption: Optional[str] = None) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
        return response.status_code == 200
    except Exception as e:
        return False

def send_telegram_video(file_path: str, caption: Optional[str] = None) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    try:
        with open(file_path, "rb") as f:
            files = {"video": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
        return response.status_code == 200
    except Exception as e:
        return False
