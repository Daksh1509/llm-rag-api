import hashlib
from cachetools import TTLCache
from typing import Optional

from app.utils.config import get_settings
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger("cache_service")


class CacheService:
    """
    In-memory TTL cache for query responses.
    Key: MD5 hash of the query string (normalized)
    Value: QueryResponse object
    TTL: Configurable via CACHE_TTL_SECONDS in .env

    For production at scale, swap TTLCache for Redis:
        import redis; r = redis.Redis(); r.setex(key, ttl, value)
    """

    def __init__(self):
        self._cache = TTLCache(
            maxsize=512,                          # Max 512 unique queries in memory
            ttl=settings.cache_ttl_seconds        # Expire after N seconds
        )

    def _make_key(self, query: str) -> str:
        """Normalize query and hash it for consistent cache lookups."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[object]:
        key = self._make_key(query)
        result = self._cache.get(key)
        if result:
            logger.debug(f"Cache HIT for key={key[:8]}...")
        return result

    def set(self, query: str, response: object) -> None:
        key = self._make_key(query)
        self._cache[key] = response
        logger.debug(f"Cached response for key={key[:8]}... (TTL={settings.cache_ttl_seconds}s)")

    def invalidate(self, query: str) -> None:
        key = self._make_key(query)
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)


cache_service = CacheService()