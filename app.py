"""
app.py — FastAPI wrapper around query.py for Rhino Bot.
"""

import os
import threading
from datetime import datetime, timedelta, timezone

import razorpay

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

# --- Razorpay client (pack payments) ---------------------------------------
_RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "").strip()
_RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
_razorpay = None
if _RAZORPAY_KEY_ID and _RAZORPAY_KEY_SECRET:
    _razorpay = razorpay.Client(auth=(_RAZORPAY_KEY_ID, _RAZORPAY_KEY_SECRET))
    print("[razorpay: configured]")
else:
    print("[razorpay: not configured - payments OFF]")

PACK_PRICE_PAISE = 19900     # ₹199, in paise (Razorpay's smallest unit)
PACK_DURATION_DAYS = 30
PACK_QUERY_LIMIT = 100       # invisible safety cap within an active pack

_FAIR_USE_MESSAGE = (
    "You've reached this pack's usage limit. Buy another 30-day pack for "
    "\u20b9199 to keep going right away."
)


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
               .select("resolved_question, answer, citations, user_name, user_email, created_at")
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
                "answer": r.get("answer") or "",
                "citations": r.get("citations") or [],
            })
            if len(items) >= 3:
                break
        return {"items": items}
    except Exception as e:
        print(f"[recent fetch failed (ignored): {e}]")
        return {"items": []}


def get_active_pack(email):
    """
    Return the user's most recent NOT-YET-EXPIRED pack row, or None. Used to
    decide whether the 15-lifetime cap should be bypassed. Fails open (returns
    None) on any DB error -- a DB blip should not revoke someone's paid access
    by accident is the wrong failure direction here, so we treat "can't tell"
    as "fall through to the free-tier check" rather than blocking outright.
    """
    if _supabase is None or not email:
        return None
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        res = (_supabase.table("packs")
               .select("id, purchased_at, expires_at")
               .eq("user_email", email)
               .gt("expires_at", now_iso)
               .order("purchased_at", desc=True)
               .limit(1)
               .execute())
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as e:
        print(f"[pack lookup failed (ignored): {e}]")
        return None


def count_user_queries_since(email, since_iso):
    """Count this user's query_logs rows since a given timestamp (for the
    invisible in-pack 100-query safety cap). Fails open (None) on DB error."""
    if _supabase is None or not email:
        return None
    try:
        res = (_supabase.table("query_logs")
               .select("id", count="exact")
               .eq("user_email", email)
               .gte("created_at", since_iso)
               .execute())
        return res.count
    except Exception as e:
        print(f"[pack usage count failed (ignored): {e}]")
        return None


class VerifyPaymentIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


@app.post("/create_order")
def create_order(request: Request):
    """Start a one-time ₹199 / 30-day pack purchase. Returns what the
    frontend needs to open Razorpay Checkout. No DB write here -- the pack is
    only granted after /verify_payment confirms a genuine signed payment."""
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"error": "Please sign in."})
    if _razorpay is None:
        return JSONResponse(status_code=500, content={"error": "Payments are not configured yet."})
    try:
        order = _razorpay.order.create({
            "amount": PACK_PRICE_PAISE,
            "currency": "INR",
            "notes": {"user_email": user.email},
        })
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key_id": _RAZORPAY_KEY_ID,
        }
    except Exception as e:
        print(f"[create_order failed: {e}]")
        return JSONResponse(status_code=500, content={"error": "Could not start payment. Try again."})


@app.post("/verify_payment")
def verify_payment(body: VerifyPaymentIn, request: Request):
    """Verify a completed payment's signature, then grant a 30-day pack.
    CRITICAL: a pack is only ever written to the DB after signature
    verification succeeds -- this is what stops a forged/tampered request
    from granting free access."""
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"error": "Please sign in."})
    if _razorpay is None:
        return JSONResponse(status_code=500, content={"error": "Payments are not configured yet."})

    try:
        _razorpay.utility.verify_payment_signature({
            "razorpay_order_id": body.razorpay_order_id,
            "razorpay_payment_id": body.razorpay_payment_id,
            "razorpay_signature": body.razorpay_signature,
        })
    except razorpay.errors.SignatureVerificationError:
        print(f"[verify_payment: BAD SIGNATURE for order {body.razorpay_order_id}]")
        return JSONResponse(status_code=400, content={"error": "Payment could not be verified."})
    except Exception as e:
        print(f"[verify_payment: error {e}]")
        return JSONResponse(status_code=400, content={"error": "Payment could not be verified."})

    # Signature is genuine -- grant the pack.
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=PACK_DURATION_DAYS)
    try:
        _supabase.table("packs").insert({
            "user_email": user.email,
            "purchased_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "razorpay_order_id": body.razorpay_order_id,
            "razorpay_payment_id": body.razorpay_payment_id,
            "amount_paise": PACK_PRICE_PAISE,
        }).execute()
    except Exception as e:
        print(f"[pack grant insert failed: {e}]")
        return JSONResponse(status_code=500, content={"error": "Payment succeeded but activation failed -- email support.artai@gmail.com with your payment id."})

    return {"ok": True, "expires_at": expires.isoformat()}


@app.get("/pack_status")
def pack_status(request: Request):
    """
    Tell the frontend whether this user currently has an active pack, so the
    persistent upgrade button in the account panel knows whether to show
    itself. Fails OPEN (active=False, i.e. show the button) on any DB error --
    a DB blip should not hide a real conversion opportunity from a free user.
    """
    user = verify_user(request)
    if user is None:
        return JSONResponse(status_code=401, content={"active": False})
    email = getattr(user, "email", None)
    if email and email.lower() in UNLIMITED:
        return {"active": True, "expires_at": None, "unlimited": True}
    pack = get_active_pack(email) if email else None
    if pack:
        return {"active": True, "expires_at": pack["expires_at"], "unlimited": False}
    return {"active": False, "expires_at": None, "unlimited": False}


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
    # request costs nothing. Unlimited emails skip it entirely. Everyone else:
    # an ACTIVE PACK bypasses the free-15 cap (subject to its own invisible
    # 100-in-30-days safety net); no active pack falls through to the
    # original free-15-lifetime check. On DB error we fail open throughout
    # rather than lock a user (paid or free) out.
    if user_email and user_email.lower() not in UNLIMITED:
        pack = get_active_pack(user_email)
        if pack:
            used_in_pack = count_user_queries_since(user_email, pack["purchased_at"])
            if used_in_pack is not None and used_in_pack >= PACK_QUERY_LIMIT:
                return {"answer": _FAIR_USE_MESSAGE, "citations": [], "grounded": False, "paywall": True}
            # else: active pack, under the safety cap -> allowed, skip free-15 check
        else:
            used = count_user_queries(user_email)
            if used is not None and used >= FREE_LIMIT:
                return {"answer": _BLOCK_MESSAGE, "citations": [], "grounded": False, "paywall": True}

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
