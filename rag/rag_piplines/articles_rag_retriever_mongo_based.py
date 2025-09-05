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
     - else: route by file_type → chunks_to_articles | apply_recency
6) chunks_to_articles*         -> collapse chunks → articles (best chunk wins) when file_type="article"
7) apply_recency               -> generic recency blend (works for chunks or articles) → final top_k

Strong defaults:
- Fusion weights: vector search 0.6 / Keyword (BM25) 0.4
- Cross-encoder: BAAI/bge-reranker-large, CE weight 0.7, CE_TOP_N 120
- Recency: half-life 45 days, weight 0.25 (0.15 if timeframe provided)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Structured output
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from typing import Optional as Opt, List as Lst

# Mongo / Embeddings / Reranker
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# ======================
# CONFIG
# ======================

MONGO_URI     = os.getenv("MONGO_URI", "mongodb+srv://...")
DB_NAME       = os.getenv("DB_NAME", "content_db")
CHUNKS_COL    = os.getenv("CHUNKS_COL", "article_chunks")
ARTICLES_COL  = os.getenv("ARTICLES_COL", "articles")

# Atlas Search config
FULLTEXT_INDEX = os.getenv("FULLTEXT_INDEX", "article_search")                  # full-text over chunks
VECTOR_INDEX   = os.getenv("VECTOR_INDEX",   "article_chuncks_vector_search")   # vector index over chunks
VECTOR_PATH    = os.getenv("VECTOR_PATH",    "dense_vector")                    # vector field on chunks

# Defaults
DEFAULT_FILE_TYPE           = os.getenv("DEFAULT_FILE_TYPE", "article")  # "article" or "chunk"
REQUESTED_K_DEFAULT         = int(os.getenv("REQUESTED_K", "10"))
FUSION_WEIGHT_VECTOR        = float(os.getenv("FUSION_WEIGHT_VECTOR", "0.6"))
FUSION_WEIGHT_TEXT          = float(os.getenv("FUSION_WEIGHT_TEXT",   "0.4"))
VEC_NUM_CANDIDATES          = int(os.getenv("VEC_NUM_CANDIDATES", "240"))
VEC_LIMIT                   = int(os.getenv("VEC_LIMIT", "240"))
FT_LIMIT                    = int(os.getenv("FT_LIMIT",  "240"))
FUSION_LIMIT                = int(os.getenv("FUSION_LIMIT", "480"))

# Cross-encoder (optional + gated)
USE_CROSS_ENCODER           = os.getenv("USE_CROSS_ENCODER", "1") not in ("0", "false", "False")
CROSS_ENCODER_MODEL         = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-large")
CE_WEIGHT                   = float(os.getenv("CE_WEIGHT", "0.7"))
CE_TOP_N                    = int(os.getenv("CE_TOP_N", "120"))   # max chunk candidates to rerank

# Gating knobs (to skip CE when unlikely to help)
CE_MIN_CANDIDATES           = int(os.getenv("CE_MIN_CANDIDATES", "40"))   # min candidates needed to justify CE
CE_MARGIN_SKIP              = float(os.getenv("CE_MARGIN_SKIP", "0.25"))  # if top1 is this much above top2 (normalized), skip CE
CE_FORCE_RERANK             = os.getenv("CE_FORCE_RERANK", "0") in ("1", "true", "True")  # force CE regardless

# Recency
RECENCY_HALF_LIFE_DAYS           = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "45"))
RECENCY_WEIGHT                   = float(os.getenv("RECENCY_WEIGHT", "0.25"))
RECENCY_WEIGHT_WITH_TIMEFILTER   = float(os.getenv("RECENCY_WEIGHT_WITH_TIMEFILTER", "0.15"))

# Embedding model + prefix config
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
_E5_LIKE = "e5" in EMBEDDING_MODEL_NAME.lower()
EMBEDDING_QUERY_PREFIX   = os.getenv("EMBEDDING_QUERY_PREFIX",   "query: "   if _E5_LIKE else "")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: " if _E5_LIKE else "")

# ======================
# STATE
# ======================

@dataclass
class SearchState:
    messages: List[BaseMessage]
    user_query: str
    file_type: str = DEFAULT_FILE_TYPE                   # "article" or "chunk"
    # extracted
    filters: Dict[str, Any] = field(default_factory=dict)     # {"author":[...], "from":"YYYY-MM-DD", "to":"YYYY-MM-DD"}
    lexical_keywords: List[str] = field(default_factory=list)
    semantic_query: str = ""
    requested_k: Optional[int] = None
    filter_warnings: List[str] = field(default_factory=list)
    # retrieval candidates (before recency)
    candidates: List[Dict[str, Any]] = field(default_factory=list)   # chunks or articles; each has "score"
    # final output
    top_results: List[Dict[str, Any]] = field(default_factory=list)


# ======================
# MODELS
# ======================

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
EMB = SentenceTransformer(EMBEDDING_MODEL_NAME) # TODO: Consider other models
CE  = CrossEncoder(CROSS_ENCODER_MODEL) if USE_CROSS_ENCODER else None # TODO: Consider other models

def embed_query(text: str) -> List[float]:
    """Embeds the query. E5-style models benefit from 'query: ' prefix; others use empty prefix."""
    return EMB.encode([f"{EMBEDDING_QUERY_PREFIX}{text}"], normalize_embeddings=True)[0].tolist()


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