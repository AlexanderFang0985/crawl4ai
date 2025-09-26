# crawler_pool.py  (new file)
import asyncio
import json
import hashlib
import time
import psutil
import logging
from contextlib import suppress
from typing import Dict

from crawl4ai import AsyncWebCrawler, BrowserConfig
from utils import load_config

CONFIG = load_config()
logger = logging.getLogger("crawler_pool")

POOL: Dict[str, AsyncWebCrawler] = {}
LAST_USED: Dict[str, float] = {}
LOCK = asyncio.Lock()

MEM_LIMIT = CONFIG.get("crawler", {}).get("memory_threshold_percent", 95.0)  # % RAM – refuse new browsers above this
IDLE_TTL = CONFIG.get("crawler", {}).get("pool", {}).get("idle_ttl_sec", 1800)  # close if unused for 30 min


def _sig(cfg: BrowserConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


async def _create_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
    await crawler.start()
    return crawler


async def _remove_crawler(sig: str, crawler: AsyncWebCrawler, *, reason: str) -> None:
    POOL.pop(sig, None)
    LAST_USED.pop(sig, None)
    logger.warning("Closing browser %s (%s)", sig[:8], reason)
    with suppress(Exception):
        await crawler.close()


async def get_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    sig = _sig(cfg)
    async with LOCK:
        existing = POOL.get(sig)
        if existing:
            if getattr(existing, "ready", False):
                LAST_USED[sig] = time.time()
                return existing
            await _remove_crawler(sig, existing, reason="stale instance")

        if psutil.virtual_memory().percent >= MEM_LIMIT:
            raise MemoryError("RAM pressure – new browser denied")

        crawler = await _create_crawler(cfg)
        POOL[sig] = crawler
        LAST_USED[sig] = time.time()
        logger.info("Started new browser %s", sig[:8])
        return crawler


async def invalidate_crawler(crawler: AsyncWebCrawler, *, reason: str = "unhealthy") -> None:
    async with LOCK:
        for sig, pooled in list(POOL.items()):
            if pooled is crawler:
                await _remove_crawler(sig, pooled, reason=reason)
                break


async def close_all():
    async with LOCK:
        for sig, crawler in list(POOL.items()):
            with suppress(Exception):
                await crawler.close()
        POOL.clear()
        LAST_USED.clear()


async def janitor():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        async with LOCK:
            for sig, crawler in list(POOL.items()):
                if now - LAST_USED.get(sig, 0) > IDLE_TTL:
                    await _remove_crawler(sig, crawler, reason="idle timeout")

