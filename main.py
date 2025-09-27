# main.py
import logging
import traceback
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from agent import SheetRAGAgent
from settings import APP_SCRIPT_URL, WEBHOOK_SECRET

# --- FastAPI setup ---
app = FastAPI(title="Sheets RAG Backend (LangGraph + LlamaIndex)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

logger = logging.getLogger("uvicorn.error")

# --- Initialize Agent ---
agent = SheetRAGAgent()


# --- Startup ---
@app.on_event("startup")
async def on_startup():
    global INDEX_BUILDING
    try:
        # Try loading persisted index
        agent.refresh(force=False)
        if agent._index is None:  # no index found
            logger.info("[Startup] No index found, scheduling background build…")
            INDEX_BUILDING = True
            asyncio.create_task(build_index_background())
        else:
            logger.info("[Startup] Index loaded successfully.")
    except Exception as e:
        logger.error(f"[Startup] Failed to init index: {e}\n{traceback.format_exc()}")

# --- Healthcheck ---
@app.get("/health")
def health():
    try:
        requests.get(APP_SCRIPT_URL, timeout=10)
        upstream = "ok"
    except Exception as e:
        upstream = f"error: {e}"
    return {"status": "ok", "apps_script": upstream}


# --- Webhook (from Apps Script) ---
@app.post("/webhook/sheets-edit")
async def webhook(req: Request):
    secret = req.headers.get("X-Webhook-Secret")
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized webhook")

    try:
        payload = await req.json()
        sheet = payload.get("sheetName")
        logger.info(f"[Webhook] Edit in sheet '{sheet}', rebuilding index…")
        agent.refresh(force=True)
        return {"status": "refreshed", "sheet": sheet}
    except Exception as e:
        logger.error(f"[Webhook ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"webhook error: {e}")


# --- Manual refresh ---
@app.post("/refresh")
def refresh():
    try:
        agent.refresh(force=True)
        return {"status": "reloaded"}
    except Exception as e:
        logger.error(f"[Refresh ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"refresh error: {e}")

@app.post("/chat")
async def chat(req: Request):
    global INDEX_BUILDING

    body = await req.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # Handle when index is not ready
    if agent._index is None:
        if INDEX_BUILDING:
            return {
                "reply": (
                    "⏳ I’m still building the knowledge index in the background. "
                    "This can take some time (around 30 minutes). "
                    "Please try again later, and I’ll be ready to answer with full context."
                )
            }
        else:
            return {
                "reply": (
                    "⚠️ The knowledge index is not available yet. "
                    "You can trigger a rebuild using `/refresh`, or wait until it is built automatically."
                )
            }

    # Normal case: index is ready
    try:
        reply = agent.chat(message)
        return {"reply": reply}
    except Exception as e:
        logger.error(f"[Chat ERROR] {e}\n{traceback.format_exc()}")
        return {
            "reply": (
                "⚠️ I ran into an error while processing your request. "
                "Please try again later."
            )
        }

# --- Debug echo (optional helper) ---
@app.post("/debug/echo")
async def echo(req: Request):
    body = await req.json()
    return {"you_sent": body}
