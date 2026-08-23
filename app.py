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

# --- Layer 3: per-user lifetime query cap ---------------------------------
# Free users get FREE_LIMIT answered queries, lifetime (no reset). The two
# emails in UNLIMITED bypass the cap entirely. Compared lowercased because
# Google can return mixed-case emails.
FREE_LIMIT = 15
UNLIMITED = {"arpitr1809@gmail.com", "artiirajpoot@gmail.com"}

# Emails hidden from the public "recently asked" feed only (NOT the cap bypass).
# Only the owner's own testing account is hidden; Arti stays visible so the feed
# isn't empty during early launch. This is separate from UNLIMITED by design.
FEED_HIDE = {"arpitr1809@gmail.com"}

_BLOCK_MESSAGE = (
    "You've used all 15 of your questions for now. "
    "RhinoBot is still in early access, so we're handling additional access "
    "personally. Email support.artai@gmail.com with \"upgrade\" in your "
    "message and we'll set you up."
)


def count_user_queries(email):
    """Count this user's lifetime rows in query_logs. On any DB error, return
    None so the caller can FAIL OPEN (a database blip must not lock users out)."""
    if _supabase is None or not email:
        return None
    try:
        res = (_supabase.table("query_logs")
               .select("id", count="exact")
               .eq("user_email", email)
               .execute())
        return res.count
    except Exception as e:
        print(f"[cap count failed, failing open: {e}]")
        return None

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


def log_query(question, answer, grounded, usage=None, user_email=None, user_name=None, citations=None, resolved_question=None):
    if _supabase is None:
        return
    try:
        row = {
            "question": question,
            "answer": answer,
            "grounded": grounded,
            "user_email": user_email,
            "user_name": user_name,
            "citations": citations,
            "resolved_question": resolved_question,
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


def _mask_name(name):
    """Backend name masking for the public recently-asked feed: show first 3
    letters of the first name, title-cased, then two stars. e.g. 'ARPIT RAJPUT'
    -> 'Dr. Arp**'. The full name never leaves the server."""
    first = ""
    if name and name.strip():
        first = name.strip().split()[0][:3].capitalize()
    return f"Dr. {first}**" if first else "Dr. Anonymous"


@app.get("/recent")
def recent(request: Request):
    """
    Recently-asked feed: the 3 most recent DISTINCT grounded questions across all
    users, shown as social proof. Excludes the unlimited (owner) emails so the
    feed reflects real user activity, not our own testing. Names are masked
    server-side. Shows resolved_question (Sonnet's standalone form) so context
    follow-ups like 'more' read as real questions. Auth-gated (after-login).
    Best-effort: on error, empty list.
    """
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"items": []})
    if _supabase is None:
        return {"items": []}
    try:
        # pull a window of recent grounded rows, then dedupe to 3 distinct questions
        res = (_supabase.table("query_logs")
               .select("resolved_question, user_name, user_email, created_at")
               .eq("grounded", True)
               .order("created_at", desc=True)
               .limit(60)
               .execute())
        rows = res.data or []
        seen = set()
        items = []
        for r in rows:
            email = (r.get("user_email") or "").lower()
            if email in FEED_HIDE:                 # hide owner testing acct from feed only
                continue
            q = (r.get("resolved_question") or "").strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:                        # distinct questions only
                continue
            seen.add(key)
            items.append({
                "who": _mask_name(r.get("user_name")),
                "question": q,
            })
            if len(items) >= 3:
                break
        return {"items": items}
    except Exception as e:
        print(f"[recent fetch failed (ignored): {e}]")
        return {"items": []}


@app.get("/worth_read")
def worth_read(request: Request):
    """
    Return the curated Worth Read items (guidelines + notable answers), newest
    first, for the right-side panel. Auth-gated like the rest of the app so only
    signed-in users see it. Only rows with active=true are returned, so an item
    can be hidden by flipping active without deleting it. Best-effort: on any DB
    error, return an empty list rather than erroring.
    """
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"items": []})
    if _supabase is None:
        return {"items": []}
    try:
        res = (_supabase.table("worth_read")
               .select("id, created_at, type, title, body, url")
               .eq("active", True)
               .order("created_at", desc=True)
               .limit(100)
               .execute())
        return {"items": res.data or []}
    except Exception as e:
        print(f"[worth_read fetch failed (ignored): {e}]")
        return {"items": []}


@app.get("/history")
def history(request: Request):
    """
    Return the logged-in user's own past turns, newest first, for the chat
    history panel. Auth-gated like /chat: a user only ever sees their own rows.
    Best-effort: on any DB error, return an empty list rather than erroring, so
    a database blip never breaks the app.
    """
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"turns": []})
    user_email = getattr(user, "email", None)
    if _supabase is None or not user_email:
        return {"turns": []}
    try:
        res = (_supabase.table("query_logs")
               .select("id, created_at, question, answer, grounded, citations")
               .eq("user_email", user_email)
               .order("created_at", desc=True)
               .limit(100)
               .execute())
        return {"turns": res.data or []}
    except Exception as e:
        print(f"[history fetch failed (ignored): {e}]")
        return {"turns": []}


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

    # --- Cap gate: checked BEFORE any retrieval/model call, so a blocked
    # request costs nothing. Unlimited emails skip it. On DB error we fail
    # open (count is None) rather than lock the user out.
    if user_email and user_email.lower() not in UNLIMITED:
        used = count_user_queries(user_email)
        if used is not None and used >= FREE_LIMIT:
            return {"answer": _BLOCK_MESSAGE, "citations": [], "grounded": False}

    msg = (body.message or "").strip()
    if not msg:
        return {"answer": "Please enter a question.", "citations": [], "grounded": False}

    with _lock:
        query.reset_usage()
        if body.mode == "general":
            text = query.sonnet_fallback(msg)
            log_query(msg, text, False, usage=query.pop_usage(),
                      user_email=user_email, user_name=user_name, citations=[],
                      resolved_question=msg)
            return {"answer": text, "citations": [], "grounded": False}

        reply, citations, grounded, resolved_q = query.answer(msg, user_id=user_id)
        log_query(msg, reply, grounded, usage=query.pop_usage(),
                  user_email=user_email, user_name=user_name, citations=citations,
                  resolved_question=resolved_q)
        return {"answer": reply, "citations": citations, "grounded": grounded}


@app.get("/{fname}")
def pwa_asset(fname: str):
    if fname in _PWA_FILES:
        path = os.path.join(_HERE, fname)
        if os.path.exists(path):
            return FileResponse(path, media_type=_PWA_FILES[fname])
    return JSONResponse(status_code=404, content={"error": "not found"})
