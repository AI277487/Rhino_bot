"""
telegram_bot.py — serve the ENT RAG over Telegram (long-polling).

Why Telegram first: no business verification, no display-name review, no SIM —
just a bot token from @BotFather. Long-polling means you need NO public URL to
test: run this on your laptop and message the bot from your phone right away.

Design (the point of the split): your RAG core stays untouched.
  query.py   -> answer(question)     # retrieve + contextualise + generate (YOURS)
  this file  -> Telegram adapter     # /start disclaimer, message -> answer() -> reply
The SAME answer() plugs into a WhatsApp webhook later; only this adapter changes.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

from query import answer            # <-- your existing RAG entry point

load_dotenv()
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ent-bot")

TG_LIMIT = 4000                      # Telegram caps at 4096; stay under with headroom

DISCLAIMER = (
    "🩺 *ENT Reference Bot*\n\n"
    "I answer ENT questions from major otolaryngology references (such as "
    "Scott-Brown and Cummings) and cite the book and page each answer draws on. "
    "I'm an educational reference, *not* medical advice\n\n"
    "Send me your question to begin."
)

# Prettify raw source labels for display. Unknown labels pass through, tidied.
SOURCE_NAMES = {
    "scott": "Scott",
    "cummings": "Cummings",
}


def _pretty_source(s):
    if not s:
        return None
    return SOURCE_NAMES.get(s, str(s).replace("_", " ").title())


def normalize_answer(result):
    """
    Adapt whatever answer() returns into (text, sources).
    Handles: a plain string; a (text, sources) tuple; or a dict with
    'text'/'answer' and 'sources'/'citations'. If yours differs, adjust HERE only.
    """
    if isinstance(result, str):
        return result, []
    if isinstance(result, tuple):
        return result[0], (result[1] if len(result) > 1 else [])
    if isinstance(result, dict):
        text = result.get("text") or result.get("answer") or ""
        sources = result.get("sources") or result.get("citations") or []
        return text, sources
    return str(result), []


def format_reply(text, sources):
    """
    Append citations that name BOTH the book and the page, e.g.
    '— Sources: Scott-Brown p.412; Cummings p.88'.
    Because we search all books at once (no routing), showing the source per
    citation is what keeps the answer honest about which book each fact came from.
    Degrades gracefully if a source is missing 'source' or 'page'.
    """
    seen, cites = set(), []
    for s in sources:
        if not isinstance(s, dict):
            continue
        src = _pretty_source(s.get("source"))
        page = s.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if src and page is not None:
            cites.append(f"{src} p.{page}")
        elif src:
            cites.append(src)
        elif page is not None:
            cites.append(f"p.{page}")
    if cites:
        return f"{text}\n\n— Sources: {'; '.join(cites)}"
    return text


def split_message(text, limit=TG_LIMIT):
    return [text[i:i + limit] for i in range(0, len(text), limit)]


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(DISCLAIMER, parse_mode="Markdown")


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    question = (update.message.text or "").strip()
    if not question:
        return
    # show "typing…" while the RAG runs
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        # run the (blocking) RAG OFF the event loop so concurrent users don't block.
        # pass the Telegram user id so each user gets their OWN conversation history.
        result = await asyncio.to_thread(answer, question, update.effective_user.id)
        text, sources = normalize_answer(result)
        reply = format_reply(text, sources) if text else \
            "I couldn't find that in my sources."
    except Exception:
        log.exception("answer() failed")
        reply = "Sorry — something went wrong answering that. Please try again."
    for chunk in split_message(reply):
        await update.message.reply_text(chunk)


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("ENT bot running (long-polling). Ctrl-C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
