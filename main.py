# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, AIMessage
# from agents.manager_agent.manager_agent import ManagerAgent
# from api_articles import router as articles_router

# load_dotenv()

# app = FastAPI(title="Noah AI News Agent", version="1.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.include_router(articles_router)


# class AskRequest(BaseModel):
#     query: str

# class AskResponse(BaseModel):
#     result: str

# @app.post("/ask", response_model=AskResponse)
# def ask_user(request: AskRequest):
    
#     user_id = "123" # TODO: Replace with actual user ID if needed using MongoDB request
#     manager = ManagerAgent(user_query=request.query, user_id=user_id)
#     messages = manager.chat()

#     last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "...")
#     return AskResponse(result=last_ai)

from typing import Any, Dict, List, Optional
from datetime import datetime
import math
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from slugify import slugify
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional

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
MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("DB_NAME")
MONGO_COLLECTION = os.getenv("COLLECTION_NAME", "articles")

if not MONGO_URI or not MONGO_DB:
    raise RuntimeError(
        "Missing Mongo settings. Please set MONGO_URI and MONGO_DB in your environment (or .env)."
    )

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
col = db[MONGO_COLLECTION]

# =============================================================================
# Schemas (Mongo -> UI mapping)
# =============================================================================
class PyObjectId(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, v: Any) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if ObjectId.is_valid(v):
            return str(v)
        raise ValueError("Invalid ObjectId")


class MongoArticle(BaseModel):
    # Store _id as a string in the model (and coerce in a validator)
    id: str = Field(alias="_id")
    url: str
    author: List[str]
    content: str
    fetched_at: Optional[str] = None
    published_at: Optional[str] = None
    section: Optional[str] = None
    source: Optional[str] = None
    title: str
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
    m = MongoArticle(**doc)

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
# Your existing /ask endpoint (kept)
# =============================================================================
from pydantic import BaseModel as PydBaseModel
from langchain_core.messages import HumanMessage, AIMessage  # noqa: F401
from agents.manager_agent.manager_agent import ManagerAgent  # keep your import


class AskRequest(PydBaseModel):
    query: str

class AskResponse(PydBaseModel):
    result: str

@app.post("/ask", response_model=AskResponse)
def ask_user(request: AskRequest):
    user_id = "123"  # TODO: replace with real user id if needed
    manager = ManagerAgent(user_query=request.query, user_id=user_id)
    messages = manager.chat()
    last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "...")
    return AskResponse(result=last_ai)
