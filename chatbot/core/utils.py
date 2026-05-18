"""
Utility helpers: cache, parsing, image conversion, excerpt/rerank, and citation extraction helper.
Keep functions lightweight and dependency minimal.
"""
import time
import hashlib
import json
import base64
import io
from PIL import Image

# simple in-memory cache for dev/demo; swap with Redis in prod
SEARCH_CACHE = {}
CACHE_TTL = 3600  # seconds

def cache_key(prefix: str, session_or_user: str, query: str) -> str:
    h = hashlib.md5(query.encode('utf-8')).hexdigest()
    return f"{prefix}:{session_or_user}:{h}"

def get_cache(key: str):
    item = SEARCH_CACHE.get(key)
    if not item:
        return None
    value, ts = item
    if time.time() - ts > CACHE_TTL:
        SEARCH_CACHE.pop(key, None)
        return None
    return value

def set_cache(key: str, value):
    SEARCH_CACHE[key] = (value, time.time())

def safe_json_parse(raw):
    """
    Tries to parse JSON-like raw input into dict.
    Returns dict (possibly {'raw_input': original}) on failure.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        return {"data": raw}
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return json.loads(s)
        except Exception:
            try:
                fixed = s.replace("'", '"').replace(",}", "}").replace(",]", "]")
                return json.loads(fixed)
            except Exception:
                return {"raw_input": raw}
    return {"raw_input": str(raw)}

def rerank_text_snippet(text: str, max_lines: int = 10) -> str:
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:max_lines])

def image_to_base64(image_path: str, max_size_px: int = 1024, jpeg_quality: int = 85):
    """
    Open image, thumbnail, convert to JPEG, return base64 string.
    """
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size_px, max_size_px))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[core.utils.image_to_base64] {e}")
        return None

def compute_file_hash(file_path: str) -> str:
    """
    MD5 of file bytes. Use for dedup checks.
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"[core.utils.compute_file_hash] {e}")
        return ""

def extract_citations(response, show_details: bool = False) -> str:
    """
    Extract simple grounding citations from Google GenAI response object.
    This expects response.candidates[0].grounding_metadata structure (may vary).
    """
    try:
        metadata = response.candidates[0].grounding_metadata
        if not (metadata and metadata.grounding_supports and metadata.grounding_chunks):
            return ""
        all_chunks = metadata.grounding_chunks
        file_citation_count = {}
        for support in metadata.grounding_supports:
            for chunk_index in support.grounding_chunk_indices:
                if 0 <= chunk_index < len(all_chunks):
                    chunk = all_chunks[chunk_index]
                    filename = getattr(chunk.retrieved_context, "title", None) or getattr(chunk.retrieved_context, "source", None) or "Unknown"
                    file_citation_count[filename] = file_citation_count.get(filename, 0) + 1
        if not file_citation_count:
            return ""
        citations_str = "\n\n--- 📚 Nguồn tham khảo ---\n"
        for fn, cnt in file_citation_count.items():
            citations_str += f"📄 {fn}" + (f" ({cnt} đoạn)" if show_details else "") + "\n"
        return citations_str
    except Exception:
        return ""
