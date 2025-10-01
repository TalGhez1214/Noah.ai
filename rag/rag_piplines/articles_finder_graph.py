"""
LangGraph Hybrid Retrieval (Atlas $rankFusion) + Optional Cross-Encoder Reranking + Recency
-------------------------------------------------------------------------------------------

Flow:
1) analyze_and_extract         -> schema-enforced extraction (filters, keywords, semantic_query, requested_k) + fallback
2) validate_filters            -> validate authors (on chunks), parse/swap dates
3) build_queries               -> lexical string for full-text
4) hybrid_retrieve             -> $rankFusion (chunks), emit chunk candidates with fused score
5) CONDITIONAL EDGE:
     - if should_use_ce(state): rerank_with_ce (Cross-Encoder) 
     - else: route by file_type → chunks_to_articles | apply_recency_rerank
6) chunks_to_articles*         -> collapse chunks → articles (best chunk wins) when file_type="article"
7) apply_recency_rerank        -> generic recency blend (works for chunks or articles) → final top_k

Strong defaults:
- Fusion weights: vector search 0.6 / Keyword (BM25) 0.4
- Cross-encoder: BAAI/bge-reranker-large, CE weight 0.7, CE_TOP_N 120
- Recency: half-life 45 days, weight 0.25 (0.15 if timeframe provided)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, time
import os

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Structured output
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional as Opt, List as Lst

# Mongo / Embeddings / Reranker
from pymongo import MongoClient
from agents.manager_agent.mongo_client import MongoClientInstance
from sentence_transformers import CrossEncoder
import numpy as np
from openai import OpenAI
from bson import ObjectId  # <-- NEW: for converting back during $match

# prompts
from rag.rag_piplines.prompts import make_analyzer_system_prompt

# ======================
# CONFIG
# ======================

MONGO_URI     = os.getenv("MONGODB_URI", "mongodb+srv://...")
DB_NAME       = os.getenv("DB_NAME", "content_db")
CHUNKS_COL    = os.getenv("CHUNKS_COL", "article_chunks")
ARTICLES_COL  = os.getenv("ARTICLES_COL", "articles")

# Atlas Search config
FULLTEXT_INDEX  = os.getenv("FULLTEXT_INDEX", "article_search")
CHUNKTEXT_INDEX = os.getenv("CHUNKTEXT_INDEX", "chunk_search")
VECTOR_INDEX    = os.getenv("VECTOR_INDEX",   "article_chuncks_vector_search")
VECTOR_PATH     = os.getenv("VECTOR_PATH",    "dense_vector")

# Defaults
DEFAULT_FILE_TYPE           = os.getenv("DEFAULT_FILE_TYPE", "article")  # "article" or "chunk"
REQUESTED_K_DEFAULT         = int(os.getenv("REQUESTED_K", "10"))
FUSION_WEIGHT_VECTOR        = float(os.getenv("FUSION_WEIGHT_VECTOR", "0.6"))
FUSION_WEIGHT_TEXT          = float(os.getenv("FUSION_WEIGHT_TEXT",   "0.4"))
RRF_K                       = int(os.getenv("RRF_K", "60"))
VEC_NUM_CANDIDATES          = int(os.getenv("VEC_NUM_CANDIDATES", "800"))
VEC_LIMIT                   = int(os.getenv("VEC_LIMIT", "80"))
FT_LIMIT                    = int(os.getenv("FT_LIMIT",  "80"))
FUSION_LIMIT                = int(os.getenv("FUSION_LIMIT", "120"))

# Cross-encoder (optional + gated)
USE_CROSS_ENCODER           = os.getenv("USE_CROSS_ENCODER", "1") not in ("0", "false", "False")
CROSS_ENCODER_MODEL         = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-large")
CE_WEIGHT                   = float(os.getenv("CE_WEIGHT", "0.7"))
CE_TOP_N                    = int(os.getenv("CE_TOP_N", "120"))

# Gating knobs
CE_MIN_CANDIDATES           = int(os.getenv("CE_MIN_CANDIDATES", "40"))
CE_MARGIN_SKIP              = float(os.getenv("CE_MARGIN_SKIP", "0.25"))

# Recency
RECENCY_HALF_LIFE_DAYS           = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "45"))
RECENCY_WEIGHT                   = float(os.getenv("RECENCY_WEIGHT", "0.25"))
RECENCY_WEIGHT_WITH_TIMEFILTER   = float(os.getenv("RECENCY_WEIGHT_WITH_TIMEFILTER", "0.15"))

# Embedding model + prefix config
OPENAI_EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072 dims

# ======================
# STATE
# ======================

@dataclass
class SearchState:
    messages: List[BaseMessage]
    user_query: str
    file_type: str = DEFAULT_FILE_TYPE
    # extracted
    filters: Dict[str, Any] = field(default_factory=dict)     # {"author":[...], "from":"YYYY-MM-DD", "to":"YYYY-MM-DD"}
    lexical_keywords: List[str] = field(default_factory=list)
    lexical_query: str = field(default="")
    semantic_query: str = ""
    requested_k: Optional[int] = None
    filter_warnings: List[str] = field(default_factory=list)
    # retrieval candidates (before recency)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    # final output
    top_results: List[Dict[str, Any]] = field(default_factory=list)
    # Mongo client (lazy init)

# ======================
# MODELS
# ======================

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CE  = CrossEncoder(CROSS_ENCODER_MODEL) if USE_CROSS_ENCODER else None

def _l2norm(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / (n + 1e-12)).tolist()

def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[text])
    return _l2norm(resp.data[0].embedding)

# ======================
# SAFE SERIALIZATION HELPERS  (NEW)
# ======================

def ensure_utc_aware(dt_or_str):
    """Coerce DB/string datetimes to tz-aware UTC; assume UTC if missing tzinfo."""
    if dt_or_str is None:
        return None
    if isinstance(dt_or_str, str):
        try:
            dt = datetime.fromisoformat(dt_or_str)
        except ValueError:
            return None
    else:
        dt = dt_or_str
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def _to_iso(dt) -> Optional[str]:
    """Coerce datetime/str/None -> ISO8601 string or None."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    try:
        return ensure_utc_aware(dt).isoformat()
    except Exception:
        return str(dt)

def _sid(x) -> Optional[str]:
    """Safe string id (ObjectId -> str)."""
    if x is None:
        return None
    try:
        return str(x)
    except Exception:
        return None

# ======================
# SCHEMA-ENFORCED EXTRACTION (+ robust fallback)
# ======================

class Filters(BaseModel):
    """Explicit, user-stated filters only. Leave null if unknown/unsure."""
    author: Opt[Lst[str]] = Field(
        default=None,
        description="Exact author names if explicitly requested. Leave null if none/unsure."
    )
    from_: Opt[str] = Field(
        default=None, alias="from",
        description="ISO date YYYY-MM-DD for earliest publication date. Leave null if none/unsure."
    )
    to: Opt[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD for latest publication date. Leave null if none/unsure."
    )

    class Config:
        allow_population_by_field_name = True  # accept 'from_' or 'from' during parsing

class Extraction(BaseModel):
    """What the retriever needs to run."""
    filters: Filters
    lexical_keywords: Lst[str] = Field(
        default_factory=list,
        description="0–12 high-value keywords/phrases for BM25 (entities, keyphrases). Leave empty if none."
    )
    semantic_query: str = Field(
        ...,
        description="Concise natural-language query for semantic retrieval. Reuse user query if already clear."
    )
    requested_k: Opt[int] = Field(
        default=None,
        description="If user asked for N results (e.g., 'give me 3 articles'), set that N. Otherwise null."
    )

    @field_validator("requested_k", mode="before")
    def clamp_requested_k(cls, v):
        if v is None:
            return v
        try:
            n = int(v)             # coerce "3" -> 3
        except (TypeError, ValueError):
            return None            # fallback to default later
        return max(1, min(50, n))  # clamp to [1, 50]


ANALYZE_SYSTEM = (
    "You extract retrieval directives for a hybrid article searcher.\n"
    "\n"
    "READ THE WHOLE CONTEXT\n"
    "- Read the entire conversation so far (user, assistant, tool messages).\n"
    "- Infer what the user wants NOW (respect references like “the first one”, “same author”, etc.).\n"
    "\n"
    "WHAT TO EXTRACT (schema fields only)\n"
    "1) filters.author → ONLY exact author names the user explicitly requested; else null.\n"
    "2) filters.from / filters.to (YYYY-MM-DD) → ONLY if the user explicitly requested a time window with concrete dates.\n"
    "   - If the user uses vague/relative time (e.g., “recent”, “lately”, “last quarter”, “last month”) and you cannot compute exact dates, leave both null.\n"
    "3) lexical_keywords → 0–12 high-value terms/phrases from the latest query (entities, key phrases). If none, return [].\n"
    "4) semantic_query → a concise natural-language query capturing the user’s intent (may reuse the user query if already clear).\n"
    "5) requested_k → number of results to return:\n"
    "   - If the user gives an exact number (e.g., “3 articles”), set that number.\n"
    "   - If the user asks for a SINGLE item with a singular noun (e.g., “an interesting article about AI”, “give me an article on …”), set 1.\n"
    "   - If the user asks plural without a number (e.g., “interesting articles about …”), set 5.\n"
    "   - If they say “a couple” → 2; “a few” → 3; “several” → 5.\n"
    "   - Otherwise, set null.\n"
    "\n"
    "STRICT RULES\n"
    "- Do NOT invent authors, dates, or numbers. If unknown/unsure → null.\n"
    "- Do NOT guess date ranges from vague language; keep them null unless concrete dates are given.\n"
    "- Keep outputs short and schema-true; no extra fields, no commentary.\n"
    "- Clamp requested_k to [1, 50] if present.\n"
    "\n"
    "OUTPUT\n"
    "- Return values EXACTLY per the schema; nothing else.\n"
)


def _messages_to_text(messages: List[BaseMessage]) -> str:
    def tag(m: BaseMessage) -> str:
        if isinstance(m, HumanMessage): return "User"
        if isinstance(m, AIMessage): return "Assistant"
        if isinstance(m, ToolMessage): return "Tool"
        if isinstance(m, SystemMessage): return "System"
        return m.__class__.__name__
    return "\n".join(f"{tag(m)}: {m.content if isinstance(m.content, str) else str(m.content)}" for m in messages)

def analyze_and_extract(state: SearchState) -> SearchState:
    structured_llm = LLM.with_structured_output(Extraction)
    transcript = _messages_to_text(state.messages)
    system_prompt = make_analyzer_system_prompt()
    prompt = [
        ("system", system_prompt),
        ("human",
         f"Conversation so far:\n{transcript}\n\n"
         f"Latest user query:\n{state.user_query}\n\n"
         "If a field is unknown or you're not sure, set it to null. Do not fabricate values.")
    ]

    try:
        result: Extraction = structured_llm.invoke(prompt)
    except ValidationError:
        result = Extraction(filters=Filters(), lexical_keywords=[], semantic_query=state.user_query, requested_k=None)
    except Exception:
        result = Extraction(filters=Filters(), lexical_keywords=[], semantic_query=state.user_query, requested_k=None)

    f: Dict[str, Any] = {}
    if result.filters.author: f["author"] = result.filters.author
    if result.filters.from_: f["from"] = result.filters.from_
    if result.filters.to:     f["to"]   = result.filters.to

    state.filters = f
    state.lexical_keywords = result.lexical_keywords or []
    state.semantic_query = result.semantic_query or state.user_query
    state.requested_k = result.requested_k
    return state

# ======================
# VALIDATE FILTERS (on CHUNK level)
# ======================

def _parse_iso_date(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def validate_filters(state: SearchState) -> SearchState:
    db = MongoClientInstance._db 
    warnings: List[str] = []

    authors = state.filters.get("author") or []
    if authors:
        existing = set(db[CHUNKS_COL].distinct("author", {"author": {"$in": authors}}))
        valid = [a for a in authors if a in existing]
        if not valid:
            warnings.append("Author filter removed (no matches on chunks).")
        state.filters["author"] = valid

    d_from = _parse_iso_date(state.filters.get("from"))
    d_to   = _parse_iso_date(state.filters.get("to"))
    if state.filters.get("from") and not d_from:
        warnings.append(f"Invalid 'from' date '{state.filters.get('from')}' removed.")
    if state.filters.get("to") and not d_to:
        warnings.append(f"Invalid 'to' date '{state.filters.get('to')}' removed.")
    if d_from and d_to and d_from > d_to:
        warnings.append("Swapped 'from' and 'to' (from > to).")
        d_from, d_to = d_to, d_from

    state.filters["from"] = d_from.date().isoformat() if d_from else None
    state.filters["to"]   = d_to.date().isoformat() if d_to else None

    state.filter_warnings = warnings
    return state

# ======================
# BUILD QUERIES
# ======================

def build_lexical_query(state: SearchState):
    return " ".join(state.lexical_keywords) if state.lexical_keywords else state.user_query

# ======================
# TZ HELPERS FOR FILTERS
# ======================

def parse_date_start_utc(date_str: str) -> datetime:
    d = datetime.fromisoformat(date_str).date()
    return datetime.combine(d, time.min, tzinfo=timezone.utc)

def parse_date_end_utc_inclusive(date_str: str) -> datetime:
    d = datetime.fromisoformat(date_str).date()
    return datetime.combine(d, time.max, tzinfo=timezone.utc)

# ======================
# HYBRID RETRIEVAL (manual RRF)
# ======================

def _vector_filter_doc(filters: Dict[str, Any]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    if filters.get("author"):
        f["author"] = {"$in": filters["author"]}
    rng = {}
    if filters.get("from"): rng["$gte"] = parse_date_start_utc(filters["from"])
    if filters.get("to"):   rng["$lte"] = parse_date_end_utc_inclusive(filters["to"])
    if rng: f["published_at"] = rng
    return f

def _search_filter_compound(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = []
    if filters.get("author"):
        f.append({
            "compound": {
                "should": [{"equals": {"path": "author", "value": a}} for a in filters["author"]],
                "minimumShouldMatch": 1
            }
        })
    rng = {}
    if filters.get("from"): rng["gte"] = parse_date_start_utc(filters["from"])
    if filters.get("to"):   rng["lte"] = parse_date_end_utc_inclusive(filters["to"])
    if rng:
        f.append({"range": {"path": "published_at", **rng}})
    return f

def _rrf_fuse(vec_docs, ft_docs, w_vec, w_ft, k):
    r_vec = {d["_id"]: i for i, d in enumerate(vec_docs)}
    r_ft  = {d["_id"]: i for i, d in enumerate(ft_docs)}
    ids = set(r_vec) | set(r_ft)
    fused = []
    for _id in ids:
        s = 0.0
        if _id in r_vec:
            s += w_vec * (1.0 / (k + r_vec[_id] + 1))
        if _id in r_ft:
            s += w_ft * (1.0 / (k + r_ft[_id] + 1))
        fused.append((_id, s))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

def hybrid_retrieve_rankfusion(state: SearchState) -> SearchState:
    chunks = MongoClientInstance._chunks

    qvec    = embed_query(state.semantic_query)
    vfilter = _vector_filter_doc(state.filters)
    sfilter = _search_filter_compound(state.filters)
    state.lexical_query = build_lexical_query(state)

    # 1) vector branch
    vec_pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": VECTOR_PATH,
            "queryVector": qvec,
            "numCandidates": VEC_NUM_CANDIDATES,
            "limit": VEC_LIMIT,
            **({"filter": vfilter} if vfilter else {})
        }},
        {"$set": {"_score_vec": {"$meta": "vectorSearchScore"}}},
        {"$project": {
            "_id": 1,
            "article_id": 1,
            "title": 1,
            "author": 1,
            "published_at": 1,
            "url": 1,
            "snippet": "$content_chunk",
            "_score_vec": 1
        }},
        {"$limit": VEC_LIMIT}
    ]

    # 2) full-text branch
    ft_pipeline = [
        {"$search": {
            "index": CHUNKTEXT_INDEX,
            "compound": {
                "filter": sfilter,
                "should": [
                    {"text": {
                        "path": ["title"],
                        "query": state.lexical_query,
                        "score": {"boost": {"value": 3}}
                    }},
                    {"text": {
                        "path": ["content_chunk"],
                        "query": state.lexical_query
                    }}
                ],
                "minimumShouldMatch": 1
            }
        }},
        {"$set": {"_score_ft": {"$meta": "searchScore"}}},
        {"$project": {
            "_id": 1,
            "article_id": 1,
            "title": 1,
            "author": 1,
            "url": 1,
            "published_at": 1,
            "snippet": "$content_chunk",
            "_score_ft": 1
        }},
        {"$limit": FT_LIMIT}
    ]

    vec_docs = list(chunks.aggregate(vec_pipeline))
    ft_docs  = list(chunks.aggregate(ft_pipeline))

    if not vec_docs and not ft_docs:
        state.candidates = []
        return state

    fused_pairs = _rrf_fuse(vec_docs, ft_docs, FUSION_WEIGHT_VECTOR, FUSION_WEIGHT_TEXT, RRF_K)

    # Build JSON-safe candidates -------------- (IDs -> str, dates -> ISO)
    by_id: Dict[Any, Dict[str, Any]] = {}
    for d in vec_docs:
        by_id[d["_id"]] = {
            "id": _sid(d["_id"]),
            "type": "chunk",
            "article_id": _sid(d.get("article_id")),
            "title": d.get("title"),
            "author": d.get("author"),
            "published_at": _to_iso(d.get("published_at")),
            "url": d.get("url"),
            "snippet": d.get("snippet"),
            "vec_score": float(d.get("_score_vec", 0.0)),
            "bm25_score": 0.0
        }
    for d in ft_docs:
        if d["_id"] in by_id:
            by_id[d["_id"]]["bm25_score"] = float(d.get("_score_ft", 0.0))
        else:
            by_id[d["_id"]] = {
                "id": _sid(d["_id"]),
                "type": "chunk",
                "article_id": _sid(d.get("article_id")),
                "title": d.get("title"),
                "author": d.get("author"),
                "published_at": _to_iso(d.get("published_at")),
                "url": d.get("url"),
                "snippet": d.get("snippet"),
                "vec_score": 0.0,
                "bm25_score": float(d.get("_score_ft", 0.0))
            }

    candidates = []
    for _id, fused_score in fused_pairs[:FUSION_LIMIT]:
        base = by_id.get(_id, {"id": _sid(_id), "type": "chunk"})
        cand = {
            **base,
            "fused_score": float(fused_score),
            "score": float(fused_score)
        }
        candidates.append(cand)

    state.candidates = candidates
    return state

# ======================
# CROSS-ENCODER RERANK
# ======================

def rerank_with_cross_encoder(state: SearchState) -> SearchState:
    if CE is None or not state.candidates:
        return state

    K = state.requested_k or REQUESTED_K_DEFAULT
    cap = min(max(K * 12, 60), CE_TOP_N)
    cand = state.candidates[:cap]

    pairs = []
    for c in cand:
        t = (c.get("title") or "")
        s = (c.get("snippet") or "")
        doc = (t + "\n" + s).strip()
        if len(doc) > 1200:
            doc = doc[:1200]
        pairs.append((state.semantic_query, doc))

    ce_scores = CE.predict(pairs)
    ce_scores = np.asarray(ce_scores, dtype=float).reshape(-1)
    fused = np.array([c["fused_score"] for c in cand], dtype=float)

    def norm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo + 1e-9) if hi > lo else np.ones_like(x) * 0.5
    n_ce, n_f = norm(ce_scores), norm(fused)
    final = CE_WEIGHT * n_ce + (1.0 - CE_WEIGHT) * n_f

    for i, c in enumerate(cand):
        c["ce_score"] = float(ce_scores[i])
        c["norm_ce"] = float(n_ce[i])
        c["norm_fused"] = float(n_f[i])
        c["score"] = float(final[i])

    cand.sort(key=lambda x: x["score"], reverse=True)
    state.candidates = cand + state.candidates[cap:]
    return state

# ======================
# CE GATING
# ======================

def _should_use_ce(state: SearchState) -> bool:
    if not USE_CROSS_ENCODER or CE is None or not state.candidates:
        return False
    if len(state.candidates) < CE_MIN_CANDIDATES:
        return False
    try:
        K = state.requested_k or REQUESTED_K_DEFAULT
        s0 = float(state.candidates[0]["score"])
        s1 = float(state.candidates[1]["score"]) if len(state.candidates) > 1 else s0
        margin = (s0 - s1) / (abs(s0) + 1e-9)
        if K <= 2 and margin >= CE_MARGIN_SKIP:
            return False
    except Exception:
        pass
    return True

# ======================
# CHUNK → ARTICLE (JSON-safe)
# ======================

def chunks_to_articles(state: SearchState) -> SearchState:
    """
    Input:  state.candidates = list of chunk candidates (each has article_id, score, etc.)
    Output: state.candidates = one entry per article (best chunk wins), type=article
    No DB calls.
    """
    if not state.candidates:
        return state

    best_by_article = {}
    for c in state.candidates:
        aid = c.get("article_id")  # stringified ObjectId already
        if not aid:
            # keep chunk as-is if no article_id (rare), or drop it:
            # best_by_article[f"__chunk__:{c.get('id')}"] = c
            continue

        cur = best_by_article.get(aid)
        if (cur is None) or (float(c.get("score", 0.0)) > float(cur.get("score", 0.0))):
            # Convert this chunk candidate into an "article" shell without the "content" field (will added later for the final requested_k)
            best_by_article[aid] = {
                "id": aid,
                "type": "article",
                "title": c.get("title"),
                "author": c.get("author"),
                "published_at": c.get("published_at"),
                "url": c.get("url"),
                "score": float(c.get("score", 0.0)),
            }

    state.candidates = list(best_by_article.values())
    return state

# ======================
# RECENCY (generic)
# ======================

def fetch_full_content_for_top_k(state: SearchState) -> SearchState:
    """
    Fetch 'content' for the first K unique article ids in state.top_results,
    and KEEP ONLY those that returned content. Order of the kept items is preserved.
    """
    if not state.top_results:
        return state

    K = state.requested_k or REQUESTED_K_DEFAULT
    top_k = state.top_results[:K]

    # Build ObjectId list in the same order (you said these are unique already)
    ids = []
    for a in top_k:
        sid = a.get("id")
        if not sid:
            continue
        try:
            ids.append(ObjectId(sid))
        except Exception:
            # bad id → skip it
            pass

    if not ids:
        # nothing to fetch; drop to empty since caller wants only-with-content
        state.top_results = []
        return state

    # Single find, fetch only _id + content
    articles_col = MongoClientInstance._articles
    docs = list(articles_col.find({"_id": {"$in": ids}}, {"_id": 1, "content": 1}))
    content_by_id = {str(d["_id"]): d.get("content") for d in docs}

    # Keep only items that have non-None content; preserve original order
    enriched = []
    for a in top_k:
        cid = a.get("id")
        content = content_by_id.get(cid)
        if content is not None:
            enriched.append({**a, "content": content})

    state.top_results = enriched
    return state

def _recency_weight_for(filters: Dict[str, Any]) -> float:
    return RECENCY_WEIGHT_WITH_TIMEFILTER if (filters.get("from") or filters.get("to")) else RECENCY_WEIGHT

def apply_recency(state: SearchState) -> SearchState:
    cands = state.candidates or []
    if not cands:
        state.top_results = []
        return state

    K = state.requested_k or REQUESTED_K_DEFAULT
    w = _recency_weight_for(state.filters)
    H = RECENCY_HALF_LIFE_DAYS

    vals = np.array([c["score"] for c in cands], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    den = (hi - lo) if hi > lo else 1.0

    now = datetime.now(timezone.utc)
    out = []
    for c in cands:
        norm = (c["score"] - lo) / den if den > 0 else 0.5
        pub = ensure_utc_aware(c.get("published_at"))
        if pub:
            age_days = max((now - pub).days, 0)
            rec = float(np.exp(- age_days / H))
        else:
            rec = 0.0
        final = (1.0 - w) * norm + w * rec
        out.append({**c, "norm_score": norm, "recency": rec, "final_score": final})

    out.sort(key=lambda x: x["final_score"], reverse=True)
    state.top_results = out[:K]

    if (state.file_type or "").strip().lower() == "article":
        state = fetch_full_content_for_top_k(state)


    return state

# ======================
# BUILD GRAPH
# ======================

def build_graph():
    g = StateGraph(SearchState)
    g.add_node("analyze_and_extract", analyze_and_extract)
    g.add_node("validate_filters",   validate_filters)
    g.add_node("hybrid_retrieve",    hybrid_retrieve_rankfusion)
    g.add_node("rerank_with_ce",     rerank_with_cross_encoder)
    g.add_node("chunks_to_articles", chunks_to_articles)
    g.add_node("apply_recency_rerank", apply_recency)

    g.set_entry_point("analyze_and_extract")
    g.add_edge("analyze_and_extract", "validate_filters")
    g.add_edge("validate_filters",   "hybrid_retrieve")

    def decide_next_after_retrieval(state: SearchState):
        if _should_use_ce(state):
            return "rerank_with_ce"
        ft = (state.file_type or DEFAULT_FILE_TYPE).strip().lower()
        return "chunks_to_articles" if ft == "article" else "apply_recency_rerank"

    g.add_conditional_edges("hybrid_retrieve", decide_next_after_retrieval, {
        "rerank_with_ce":     "rerank_with_ce",
        "chunks_to_articles": "chunks_to_articles",
        "apply_recency_rerank":"apply_recency_rerank"
    })

    def route_after_ce(state: SearchState):
        ft = (state.file_type or DEFAULT_FILE_TYPE).strip().lower()
        return "chunks_to_articles" if ft == "article" else "apply_recency_rerank"

    g.add_conditional_edges("rerank_with_ce", route_after_ce, {
        "chunks_to_articles": "chunks_to_articles",
        "apply_recency_rerank":"apply_recency_rerank"
    })

    g.add_edge("chunks_to_articles", "apply_recency_rerank")
    g.add_edge("apply_recency_rerank", END)
    return g.compile()
