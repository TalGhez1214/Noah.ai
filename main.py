from typing import Any, Dict, List, Optional
from datetime import datetime
import math
import os
from urllib.parse import urlparse
from datetime import datetime, date, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from slugify import slugify

import asyncio
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
import json
from agents.inline_agents.explainer import ExplainerAgent
from typing import AsyncIterator

from agents.inline_agents.asker import AskerAgent


# --- Load env (.env) ---
load_dotenv()

# =============================================================================
# App
# =============================================================================
app = FastAPI(title="Noah AI News Agent", version="1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MongoDB
# =============================================================================
# NOTE: using your env var names here
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("DB_NAME")
MONGO_COLLECTION = os.getenv("COLLECTION_NAME", "articles")

if not MONGO_URI or not MONGO_DB:
    raise RuntimeError(
        "Missing Mongo settings. Please set MONGODB_URI and DB_NAME in your environment (or .env)."
    )

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
col = db[MONGO_COLLECTION]

# =============================================================================
# Schemas (Mongo -> UI mapping)
# =============================================================================
class MongoArticle(BaseModel):
    # Store _id as a string in the model (and coerce in a validator)
    id: str = Field(alias="_id")
    url: str = ""
    author: List[str] = Field(default_factory=list)
    content: str = ""
    fetched_at: Optional[str] = None
    published_at: Optional[str] = None
    section: Optional[str] = None
    source: Optional[str] = None
    title: str = ""
    topic: Optional[str] = None
    readingTime: Optional[str] = None  # e.g., "6 min"
    coverImage: Optional[str] = None

    # --- validators (Pydantic v2) ---
    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        # Handle bson.ObjectId or anything else
        return str(v)

    @field_validator("author", mode="before")
    @classmethod
    def _author_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        # came as a single string or something else
        return [str(v)]


class UIAuthor(BaseModel):
    name: str
    avatar: str


class UIArticle(BaseModel):
    id: str
    slug: str
    title: str
    excerpt: str
    content: str
    author: UIAuthor
    publishedAt: str
    category: str
    tags: List[str]
    readingTime: int
    coverImage: str
    featured: bool = False


def _estimate_reading_minutes(text: str) -> int:
    words = max(1, len((text or "").split()))
    return max(1, math.ceil(words / 220))  # ~220 wpm


def _to_ui_article(doc: Dict[str, Any]) -> UIArticle:
    d = dict(doc)
    for k in ("published_at", "fetched_at"):
        v = d.get(k)
        if isinstance(v, datetime):
            d[k] = (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
        elif isinstance(v, date):
            d[k] = datetime(v.year, v.month, v.day, tzinfo=timezone.utc).isoformat()

    m = MongoArticle(**d)

    slug_val = slugify(m.title) if m.title else str(m.id)

    # choose a published timestamp (prefer published_at, then fetched_at, else now)
    iso_published = m.published_at or m.fetched_at or (datetime.utcnow().isoformat() + "Z")

    # excerpt from first ~220 chars
    excerpt_val = (m.content or "").strip().replace("\n", " ")
    if len(excerpt_val) > 220:
        excerpt_val = excerpt_val[:220] + "…"

    # normalize readingTime to int minutes
    if m.readingTime and isinstance(m.readingTime, str) and m.readingTime.strip().endswith("min"):
        try:
            minutes = int(m.readingTime.strip().split()[0])
        except Exception:
            minutes = _estimate_reading_minutes(m.content or "")
    else:
        minutes = _estimate_reading_minutes(m.content or "")

    # ✅ seed tags with the topic if present
    tags: List[str] = []
    if m.topic:
        tags.append(m.topic)

    return UIArticle(
        id=str(m.id),
        slug=slug_val,
        title=m.title,
        excerpt=excerpt_val,
        content=m.content,
        author=UIAuthor(
            name=(m.author[0] if m.author else "Unknown"),
            avatar="https://picsum.photos/100/100?random=42",
        ),
        publishedAt=iso_published,
        category=m.topic or "General",
        tags=tags,
        readingTime=minutes,
        coverImage=m.coverImage or "https://picsum.photos/800/400?random=101",
        featured=False,
    )

# =============================================================================
# Articles Endpoints
# =============================================================================
@app.get("/health")
async def health():
    doc = await col.find_one({}, {"_id": 1})
    return {"ok": True, "hasData": bool(doc)}

@app.get("/articles", response_model=List[UIArticle])
async def list_articles(
    q: Optional[str] = Query(None, description="Full-text query over title/content"),
    topic: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    filt: Dict[str, Any] = {}
    if topic:
        filt["topic"] = topic
    if q:
        filt["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"content": {"$regex": q, "$options": "i"}},
        ]

    cursor = col.find(filt).sort("published_at", -1).skip(skip).limit(limit)
    docs = [_to_ui_article(doc) async for doc in cursor]
    return docs

@app.get("/articles/{article_id}", response_model=UIArticle)
async def get_article(article_id: str):
    try:
        oid = ObjectId(article_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid article id")
    doc = await col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")
    return _to_ui_article(doc)

@app.get("/articles/by-slug/{slug}", response_model=UIArticle)
async def get_article_by_slug(slug: str):
    # First pass: fetch titles only for quick slug match
    projection = {"title": 1}
    async for d in col.find({}, projection):
        if slugify(d.get("title", "")) == slug:
            full = await col.find_one({"_id": d["_id"]})
            return _to_ui_article(full)
    raise HTTPException(status_code=404, detail="Article not found")

# =============================================================================
# /ask endpoint — now page-aware
# =============================================================================
from langchain_core.messages import HumanMessage, AIMessage  # noqa: F401
from agents.manager_agent.manager_agent import ManagerAgent  # keep your import

class AskRequest(BaseModel):
    query: str
    page_url: Optional[str] = None 

class AskResponse(BaseModel):
    result: str

from urllib.parse import urlparse, unquote


from datetime import datetime, date
from bson import ObjectId

def _sanitize_for_state(value):
    """Recursively convert Mongo / Python objects into msgpack-safe types."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        # ISO8601 string
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return {str(k): _sanitize_for_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_state(v) for v in value)
    # numbers, strings, bool, None are fine
    return value

def _slim_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """(Optional) keep only fields your agent actually needs."""
    if not doc:
        return None
    return {
        "_id": doc.get("_id"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "topic": doc.get("topic"),
        "source": doc.get("source"),
        "published_at": doc.get("published_at"),
        "fetched_at": doc.get("fetched_at"),
        "section": doc.get("section"),
        "coverImage": doc.get("coverImage"),
        "author": doc.get("author"),
        "content": doc.get("content"),
        "readingTime": doc.get("readingTime"),
    }

async def _find_doc_for_page(page_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Resolve the current page's Mongo document.
    A) If URL looks like your app's /articles/<slug>, match by slugified title.
    B) Else try direct match by the stored source URL: { url: page_url }.
    """
    if not page_url:
        return None

    try:
        parsed = urlparse(page_url)
    except Exception:
        return None

    # Normalize path: strip leading/trailing slashes, drop any trailing slash, decode
    raw_path = (parsed.path or "").strip("/")
    path = unquote(raw_path)
    parts = [p for p in path.split("/") if p]

    # Case A: app route /articles/<slug>
    if len(parts) >= 2 and parts[0].lower() == "articles":
        slug = parts[1].lower().strip()
        # Fast projection pass: compare slugified titles in Python (no DB scan of full docs)
        projection = {"title": 1}
        async for d in col.find({}, projection):
            title = d.get("title", "")
            if title and slugify(title).lower() == slug:
                return await col.find_one({"_id": d["_id"]})

        # Fallback (best-effort): try regex on title with all slug tokens present
        # e.g., "saas companies unbundle ai to fix margins and calm renewals"
        tokens = [t for t in slug.replace("-", " ").split() if t]
        if tokens:
            and_regex = [{"title": {"$regex": t, "$options": "i"}} for t in tokens]
            candidate = await col.find_one({"$and": and_regex}, {"title": 1})
            if candidate:
                return await col.find_one({"_id": candidate["_id"]})

    # Case B: try exact URL match to the source URL in Mongo (for off-site originals)
    doc = await col.find_one({"url": page_url})
    if doc:
        return doc

    return None

@app.post("/ask", response_model=AskResponse)
async def ask_user(request: AskRequest):
    """
    Accepts:
      - query: the user question
      - page_url: the current page URL from the client

    Looks up the matching article in Mongo and injects it into the manager's state
    as 'current_page'.
    """
    # Resolve current page document (or None)
    current_doc = await _find_doc_for_page(request.page_url)

    # (optional) reduce payload to what's needed
    current_doc = _slim_doc(current_doc)

    # ✅ sanitize recursively so langgraph/msgpack can serialize the state
    current_doc = _sanitize_for_state(current_doc)

    user_id = "123"  # TODO: replace with real user id if needed
    manager = ManagerAgent(
        user_query=request.query,
        user_id=user_id,
        current_page=current_doc, 
    )

    messages = manager.chat()
    messages.pop() # TODO remove this

    last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "...")
    return AskResponse(result=last_ai)


def ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    current_doc = await _find_doc_for_page(req.page_url)
    current_doc = _sanitize_for_state(_slim_doc(current_doc))
    user_id = "123"

    async def gen():
        
        manager = ManagerAgent(
            user_query=req.query,
            user_id=user_id,
            current_page=current_doc,
            use_checkpointer=True,
        )

        last_state = None

        async for event in manager.app.astream_events(
            {
                "messages": [HumanMessage(content=req.query)],
                "user_query": req.query,
                "agent": None,
                "current_page": current_doc,
            },
            config={"configurable": {"thread_id": user_id}},
            version="v2",
        ):
            ev = event.get("event")
            data = event.get("data", {})
            tags = event.get("tags", []) or []

            # ✅ STREAM ONLY SUPERVISOR TEXT
            if ev == "on_chat_model_stream" and "node:supervisor" in tags:
                ch = data.get("chunk")
                if ch and getattr(ch, "content", ""):
                    yield ndjson({"type": "token", "data": ch.content})

            # track a possible final state when it appears
            if ev in ("on_chain_end", "on_graph_end", "on_tool_end"):
                out = data.get("output")
                if isinstance(out, dict):
                    last_state = out

        # Fallback: if we didn't capture state from events, run once to get it
        if last_state is None:
            last_state = await manager.app.ainvoke(
                {
                    "messages": [HumanMessage(content=req.query)],
                    "user_query": req.query,
                    "agent": None,
                    "current_page": current_doc,
                },
                config={"configurable": {"thread_id": user_id}},
            )
        
        msgs = last_state.get("messages", [])
        print("\n🗨️ Full conversation (including tool messages):")
        for m in msgs:
            try:
                if hasattr(m, "pretty_print"):
                    m.pretty_print()          # LangChain-native messages
                else:
                    print("TOOL/RAW:", m)      # Fallback for dicts / custom payloads
            except Exception as e:
                print("<<could not render message>>", type(m), m, e)

        # Send non-streamed modals (or any UI payload you carry)
        if isinstance(last_state, dict) and last_state.get("ui_items") is not None:
            yield ndjson({"type": "ui_items", "data": last_state["ui_items"]})

        yield ndjson({"type": "done"})

    return StreamingResponse(gen(), media_type="application/x-ndjson")



class InlineExplainRequest(BaseModel):
    highlighted_text: str
    page_url: Optional[str] = None

class InlineExplainResponse(BaseModel):
    result: str


from fastapi.responses import StreamingResponse
from typing import AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage
from agents.inline_agents.explainer import ExplainerAgent
from pydantic import BaseModel

# @app.post("/inline_explain")
# async def inline_explain(req: InlineExplainRequest):
#     """
#     Streams ONLY text tokens (no JSON framing), and pretty-prints the full
#     conversation (like ask_stream) after completion.
#     """
#     # Resolve page content via your helper (same style as ask_stream)
#     current_doc = await _find_doc_for_page(req.page_url)

#     # Try to extract content (keep this permissive; no other changes)
#     page_content = (
#         (current_doc.get("content")
#          or current_doc.get("page_content")
#          or current_doc.get("text")
#          or current_doc.get("body")
#          or "")
#         if current_doc else ""
#     ).strip()

#     explainer = ExplainerAgent(model="gpt-4o-mini", temperature=0.2)

#     human = HumanMessage(content="Explain the highlighted selection in context.")
#     state = {"messages": [human]}
#     config = {
#         "configurable": {
#             "thread_id": "inline-explain",
#             "current_page_content": page_content,
#             "highlighted_text": req.highlighted_text,
#             "today": __import__("datetime").date.today().isoformat(),
#         }
#     }

#     async def event_stream() -> AsyncIterator[bytes]:
#         last_state = None

#         # Stream ONLY chat model token chunks as plain text
#         async for ev in explainer.app.astream_events(
#             state,
#             config=config,
#             version="v2",
#         ):
#             event_name = ev.get("event")
#             data = ev.get("data", {})

#             if event_name == "on_chat_model_stream":
#                 ch = data.get("chunk")
#                 if ch:
#                     try:
#                         text = "".join(getattr(c, "text", "") for c in getattr(ch, "content", []))
#                     except Exception:
#                         text = getattr(ch, "content", "") or getattr(ch, "text", "") or ""
#                     if text:
#                         yield text.encode("utf-8")

#             # Track final state when available (for pretty print)
#             if event_name in ("on_chain_end", "on_graph_end", "on_tool_end"):
#                 out = data.get("output")
#                 if isinstance(out, dict):
#                     last_state = out

#         # Fallback: if we didn’t capture state from events, run once to get it
#         if last_state is None:
#             last_state = await explainer.app.ainvoke(state, config=config)

#         # === Pretty print (identical style to your ask_stream) ===
#         msgs = (last_state or {}).get("messages", [])
#         print("\n🗨️ Full conversation (including tool messages):")
#         for m in msgs:
#             try:
#                 if hasattr(m, "pretty_print"):
#                     m.pretty_print()           # LangChain-native messages
#                 else:
#                     print("TOOL/RAW:", m)       # Fallback for dicts / custom payloads
#             except Exception as e:
#                 print("<<could not render message>>", type(m), m, e)

#         # Final newline so clients know we’re done
#         yield b"\n"

#     return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")








# @app.post("/inline_explain")
# async def inline_explain(req: InlineExplainRequest):
#     """
#     Streams ONLY text tokens (no JSON framing), and pretty-prints the full
#     conversation after completion (same style as ask_stream).
#     """
#     # Resolve current doc via your helper (like ask_stream)
#     current_doc = await _find_doc_for_page(req.page_url)
#     page_content = current_doc.get("content", "") or ""

#     explainer = ExplainerAgent(model="gpt-4o-mini", temperature=0.2)

#     # ✅ Put the selection directly in the user message as a fallback
#     human = HumanMessage(
#         content=f'Explain this selection in the context of the current page:\n""" {req.highlighted_text} """'
#     )
#     state = {"messages": [human]}
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(page_content)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#     # ✅ And pass it via config (what your prompt actually reads)
#     config = {
#         "configurable": {
#             "thread_id": "inline-explain",
#             "current_page_content": page_content,
#             "highlighted_text": req.highlighted_text,   # <-- IMPORTANT
#             "today": __import__("datetime").date.today().isoformat(),
#         }
#     }

#     # Quick debug so you can see it's coming through
#     #print("[inline_explain] highlighted_len:", len(req.highlighted_text), "| page_has_content:", bool(page_content))

#     async def event_stream():
#         last_state = None

#         # ✅ Use v2 like your ask_stream (v1 won’t emit the token event you’re handling)
#         async for ev in explainer.app.astream_events(
#             state,
#             config=config,
#             version="v2",
#         ):
#             event_name = ev.get("event")
#             data = ev.get("data", {})

#             if event_name == "on_chat_model_stream":
#                 ch = data.get("chunk")
#                 if ch:
#                     try:
#                         text = "".join(getattr(c, "text", "") for c in getattr(ch, "content", []))
#                     except Exception:
#                         text = getattr(ch, "content", "") or getattr(ch, "text", "") or ""
#                     if text:
#                         yield text.encode("utf-8")

#             # Track final graph/chain/tool output so we can pretty-print after
#             if event_name in ("on_chain_end", "on_graph_end", "on_tool_end"):
#                 out = data.get("output")
#                 if isinstance(out, dict):
#                     last_state = out

#         # Fallback: fetch final state if we didn’t capture it during streaming
#         if last_state is None:
#             last_state = await explainer.app.ainvoke(state, config=config)

#         # === Pretty print (same pattern you use in ask_stream) ===
#         msgs = (last_state or {}).get("messages", [])
#         print("\n🗨️ Full conversation (including tool messages):")
#         for m in msgs:
#             try:
#                 if hasattr(m, "pretty_print"):
#                     m.pretty_print()
#                 else:
#                     print("TOOL/RAW:", m)
#             except Exception as e:
#                 print("<<could not render message>>", type(m), m, e)

#         yield b"\n"

#     return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")



@app.post("/inline_explain", response_model=InlineExplainResponse)
async def inline_explain(req: InlineExplainRequest):
    # get the page doc like you do elsewhere
    current_doc = await _find_doc_for_page(req.page_url)
    page_content = (
        (current_doc.get("content")
         or current_doc.get("page_content")
         or current_doc.get("text")
         or current_doc.get("body")
         or "")
        if current_doc else ""
    ).strip()

    explainer = ExplainerAgent(model="gpt-4o-mini", temperature=0.2)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("[inline_explain] highlighted_len:", req.highlighted_text, "| page_has_content:", page_content)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # invoke via the agent's call() (no astream_events)
    msgs = explainer.call(
        highlighted_text=req.highlighted_text,
        current_page_content=page_content,
        thread_id="inline-explain",
    )

    # pretty print (kept, since you liked it)
    print("\n🗨️ Full conversation (including tool messages):")
    for m in msgs:
        try:
            if hasattr(m, "pretty_print"):
                m.pretty_print()
            else:
                print("TOOL/RAW:", m)
        except Exception as e:
            print("<<could not render message>>", type(m), m, e)

    last_ai = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage)), "")
    return InlineExplainResponse(result=last_ai or "Sorry—I couldn’t generate an explanation.")

class InlineAskRequest(BaseModel):
    highlighted_text: str
    page_url: str
    user_query: str

class InlineAskResponse(BaseModel):
    result: str


@app.post("/inline_ask", response_model=InlineAskResponse)
async def inline_ask(req: InlineAskRequest):
    # Resolve the page doc (same as inline_explain)
    current_doc = await _find_doc_for_page(req.page_url)
    page_content = (
        (current_doc.get("content")
         or current_doc.get("page_content")
         or current_doc.get("text")
         or current_doc.get("body")
         or "")
        if current_doc else ""
    ).strip()

    asker = AskerAgent(model="gpt-4o-mini", temperature=0.2)

    # tiny debug
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("[inline_ask] user_query:", req.user_query)
    print("[inline_ask] highlighted_len:", req.highlighted_text, "| page_has_content:", page_content)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # invoke via the agent's call()
    msgs = asker.call(
        user_query=req.user_query,
        highlighted_text=req.highlighted_text,
        current_page_content=page_content,
        thread_id="inline-ask",
    )

    # pretty print (same style as explain)
    print("\n🗨️ Full conversation (including tool messages):")
    for m in msgs:
        try:
            if hasattr(m, "pretty_print"):
                m.pretty_print()
            else:
                print("TOOL/RAW:", m)
        except Exception as e:
            print("<<could not render message>>", type(m), m, e)

    last_ai = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage)), "")
    return InlineAskResponse(result=last_ai or "Sorry—I couldn’t generate an answer.")