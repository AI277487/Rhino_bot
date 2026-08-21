"""
app.py — FastAPI wrapper around query.py for Rhino Bot.
"""

import os
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import query  # noqa: E402

app = FastAPI(title="Rhino Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "rhino-bot-ui.html")

_ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()

_lock = threading.Lock()

# --- Supabase query logging (Layer 1) -------------------------------------
_supabase = None
try:
    _SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    _SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if _SUPABASE_URL and _SUPABASE_KEY:
        from supabase import create_client
        _supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        print("[supabase: connected - query logging ON]")
    else:
        print("[supabase: not configured - query logging OFF]")
except Exception as e:
    print(f"[supabase: init failed, logging OFF - {e}]")


def log_query(question, answer, grounded, usage=None, user_email=None, user_name=None):
    if _supabase is None:
        return
    try:
        row = {
            "question": question,
            "answer": answer,
            "grounded": grounded,
            "user_email": user_email,
            "user_name": user_name,
        }
        if usage:
            row.update({
                "input_tokens":   usage["input_tokens"],
                "output_tokens":  usage["output_tokens"],
                "total_tokens":   usage["total_tokens"],
                "cost_usd":       usage["cost_usd"],
                "cost_breakdown": usage["cost_breakdown"],
            })
        _supabase.table("query_logs").insert(row).execute()
    except Exception as e:
        print(f"[supabase log failed (ignored): {e}]")


def verify_user(request):
    if _supabase is None:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        res = _supabase.auth.get_user(token)
        return res.user if res and res.user else None
    except Exception as e:
        print(f"[token verify failed: {e}]")
        return None


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = "grounded"
    passcode: str = ""


@app.get("/")
def index():
    return FileResponse(_INDEX)


@app.get("/privacy")
def privacy():
    return FileResponse(os.path.join(_HERE, "privacy.html"))


_PWA_FILES = {
    "manifest.webmanifest": "application/manifest+json",
    "sw.js": "application/javascript",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "icon-maskable-512.png": "image/png",
    "apple-touch-icon.png": "image/png",
    "favicon-48.png": "image/png",
}


@app.get("/health")
def health():
    return {"ok": True, "chunks": query.collection.count()}


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    user = verify_user(request)
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"answer": "Please sign in to use RhinoBot.",
                     "citations": [], "grounded": False},
        )
    user_id = user.id
    user_email = getattr(user, "email", None)
    _meta = getattr(user, "user_metadata", None) or {}
    user_name = _meta.get("full_name") or _meta.get("name") or None

    msg = (body.message or "").strip()
    if not msg:
        return {"answer": "Please enter a question.", "citations": [], "grounded": False}

    with _lock:
        query.reset_usage()
        if body.mode == "general":
            text = query.sonnet_fallback(msg)
            log_query(msg, text, False, usage=query.pop_usage(),
                      user_email=user_email, user_name=user_name)
            return {"answer": text, "citations": [], "grounded": False}

        reply, citations, grounded = query.answer(msg, user_id=user_id)
        log_query(msg, reply, grounded, usage=query.pop_usage(),
                  user_email=user_email, user_name=user_name)
        return {"answer": reply, "citations": citations, "grounded": grounded}


@app.get("/{fname}")
def pwa_asset(fname: str):
    if fname in _PWA_FILES:
        path = os.path.join(_HERE, fname)
        if os.path.exists(path):
            return FileResponse(path, media_type=_PWA_FILES[fname])
    return JSONResponse(status_code=404, content={"error": "not found"})
