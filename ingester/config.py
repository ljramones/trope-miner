#!/usr/bin/env python3
import os
from dataclasses import dataclass

@dataclass
class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    REASONER_MODEL:  str = os.getenv("REASONER_MODEL", "llama3.1:8b")
    EMBED_MODEL:     str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    CHROMA_HOST:     str = os.getenv("CHROMA_HOST", "127.0.0.1")
    CHROMA_PORT:     int = int(os.getenv("CHROMA_PORT", "8000"))
    CHUNK_COLLECTION:str = os.getenv("CHUNK_COLLECTION", "trope-miner-v1-cos")

    RERANK_TOP_K:    int = int(os.getenv("RERANK_TOP_K", "8"))
    RERANK_KEEP_M:   int = int(os.getenv("RERANK_KEEP_M", "3"))
    DOC_CHAR_MAX:    int = int(os.getenv("RERANK_DOC_CHAR_MAX", "480"))

    DOWNWEIGHT_NO_MENTION: float = float(os.getenv("DOWNWEIGHT_NO_MENTION", "0.55"))
    SEM_SIM_THRESHOLD:     float = float(os.getenv("SEM_SIM_THRESHOLD", "0.36"))

def env_from_cli(**kwargs):
    """Mirror CLI args into env so modules that read os.getenv() stay in sync."""
    for k, v in kwargs.items():
        if v is None:
            continue
        os.environ[k] = str(v)
