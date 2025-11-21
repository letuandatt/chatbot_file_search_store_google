import time
import hashlib
import redis

from abc import ABC, abstractmethod
from chatbot.config import config as app_config


class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str): pass

    @abstractmethod
    def set(self, key: str, value, ttl: int = 3600): pass

    @staticmethod
    def generate_key(prefix: str, *args) -> str:
        raw = f"{prefix}:" + ":".join(str(arg) for arg in args)
        return hashlib.md5(raw.encode('utf-8')).hexdigest()


class RedisCache(BaseCache):
    def __init__(self, redis_url: str):
        if redis is None:
            raise ImportError("Thiếu thư viện redis. Cài đặt: pip install redis")
        # decode_responses=True để tự convert bytes -> string
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.client.ping()  # Test connection
        print(f"[Cache] Connected to Redis: {redis_url}")

    def get(self, key: str):
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value, ttl: int = 3600):
        try:
            if not isinstance(value, str): value = str(value)
            self.client.setex(name=key, time=ttl, value=value)
        except Exception as e:
            print(f"[Cache] Set error: {e}")


def init_cache():
    return RedisCache(app_config.REDIS_URL)


# Singleton
app_cache = init_cache()
