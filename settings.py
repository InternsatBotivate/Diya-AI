# settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Required ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
APP_SCRIPT_URL = os.getenv("APP_SCRIPT_URL", "").strip()  # your Apps Script /exec URL
WEBHOOK_SECRET  = os.getenv("WEBHOOK_SECRET", "").strip()  # same as in Apps Script

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not APP_SCRIPT_URL:
    raise RuntimeError("APP_SCRIPT_URL missing")
if not WEBHOOK_SECRET:
    raise RuntimeError("WEBHOOK_SECRET missing")

# --- Optional ---
MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage").strip()
os.makedirs(PERSIST_DIR, exist_ok=True)
