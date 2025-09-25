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
    page_url: Optional[str] = None  # ✅ NEW: current page URL from the client

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

    last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "...")
    return AskResponse(result=last_ai)
