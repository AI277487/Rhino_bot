"""
app.py — FastAPI wrapper around query.py for Rhino Bot.

Serves:
  GET  /       -> the single-file React frontend (rhino-bot-ui.html)
  POST /chat   -> {message, session_id, mode} -> {answer, citations, grounded}

Importing `query` runs its startup once (loads ChromaDB, builds the BM25
index), so the first request after boot is fast. Run with:

    uvicorn app:app --host 127.0.0.1 --port 8000

and put HTTPS in front of it (see deploy notes).
"""

import os
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Load .env from THIS file's own folder, by absolute path, BEFORE anything
# reads an environment variable. systemd may run the app from a different
# working directory, so a bare load_dotenv() can silently find nothing —
# pinning the path is what makes SUPABASE_URL / SUPABASE_SERVICE_KEY reliable.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import query  # noqa: E402  (import triggers Chroma load + BM25 build, intentional)

app = FastAPI(title="Rhino Bot API")

# Same-origin in the demo (frontend served below), so CORS isn't strictly
# needed — kept permissive in case you host the HTML elsewhere. Tighten for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.join(_HERE, "rhino-bot-ui.html")

# Shared demo password, read from .env (ACCESS_PASSWORD=...). If it's unset,
# the gate is OFF (app answers freely) — so setting it in .env is what turns
# protection on. Keep it out of the code so it's easy to rotate.
_ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()

# query.py uses module-global state (histories, embedder, BM25). FastAPI runs
# sync endpoints in a threadpool, so serialize the pipeline to avoid two
# requests racing on that shared state. Fine at demo concurrency; revisit if
# you need real parallelism.
_lock = threading.Lock()

# --- Supabase query logging (Layer 1) -------------------------------------
# Connect using the SERVER-ONLY secret key from .env. If either value is
# missing, logging is silently disabled so the app still runs normally.
_supabase = None
try:
    _SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
    _SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if _SUPABASE_URL and _SUPABASE_KEY:
        from supabase import create_client
        _supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        print("[supabase: connected — query logging ON]")
    else:
        print("[supabase: not configured — query logging OFF]")
except Exception as e:
    print(f"[supabase: init failed, logging OFF — {e}]")


def log_query(question, answer, grounded):
    """
    Write one question/answer to the query_logs table. Best-effort only:
    logging must NEVER break answering, so any failure is swallowed. The user
    always gets their answer even if the database is down.
    """
    if _supabase is None:
        return
    try:
        _supabase.table("query_logs").insert({
            "question": question,
            "answer": answer,
            "grounded": grounded,
        }).execute()
    except Exception as e:
        print(f"[supabase log failed (ignored): {e}]")


def verify_user(request):
    """
    Read the 'Authorization: Bearer <token>' header, ask Supabase whether the
    token is genuine and unexpired, and return the user (with .id and .email)
    if so — otherwise None. This is the server-side gate: no valid token, no
    user, no answer. Uses Supabase's own validation, so we never trust the
    token blindly.
    """
    if _supabase is None:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        # ask Supabase to validate the token and tell us who it belongs to
        res = _supabase.auth.get_user(token)
        return res.user if res and res.user else None
    except Exception as e:
        print(f"[token verify failed: {e}]")
        return None


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = "grounded"      # "grounded" (RAG) | "general" (Sonnet fallback)
    passcode: str = ""          # shared demo password sent by the frontend


@app.get("/")
def index():
    return FileResponse(_INDEX)


@app.get("/privacy")
def privacy():
    return FileResponse(os.path.join(_HERE, "privacy.html"))


# --- PWA static files (served from the same folder as the HTML) ---
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
    # --- Auth gate: require a valid Supabase login token ---
    # Checked BEFORE any retrieval or model call, so an unauthenticated
    # request costs $0. This is the real server-side gate — the login screen
    # on the frontend is not enough on its own.
    user = verify_user(request)
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"answer": "Please sign in to use RhinoBot.",
                     "citations": [], "grounded": False},
        )
    user_id = user.id          # stable per-user id, used for logging and (next) caps
    user_email = getattr(user, "email", None)

    msg = (body.message or "").strip()
    if not msg:
        return {"answer": "Please enter a question.", "citations": [], "grounded": False}

    with _lock:
        if body.mode == "general":
            # the frontend's "get general answer" button
            text = query.sonnet_fallback(msg)
            log_query(msg, text, False)
            return {"answer": text, "citations": [], "grounded": False}

        reply, citations, grounded = query.answer(msg, user_id=user_id)
        log_query(msg, reply, grounded)
        return {"answer": reply, "citations": citations, "grounded": grounded}


# --- PWA asset catch-all: MUST be last so it never shadows /health or /chat ---
@app.get("/{fname}")
def pwa_asset(fname: str):
    if fname in _PWA_FILES:
        path = os.path.join(_HERE, fname)
        if os.path.exists(path):
            return FileResponse(path, media_type=_PWA_FILES[fname])
    return JSONResponse(status_code=404, content={"error": "not found"})
