# crawler_pool.py - checkout/return crawler pool (one request per crawler)
import asyncio
import json
import hashlib
import time
from collections import deque
from contextlib import suppress
from typing import Dict, Deque, Set, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig
from utils import load_config, get_container_memory_percent
import logging

logger = logging.getLogger(__name__)
CONFIG = load_config()

MEM_LIMIT = CONFIG.get("crawler", {}).get("memory_threshold_percent", 95.0)
IDLE_TTL = CONFIG.get("crawler", {}).get("pool", {}).get("idle_ttl_sec", 1800)
POOL_MAX = CONFIG.get("crawler", {}).get("pool", {}).get("max_browsers", 10)


class _Pool:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.total = 0
        self.available: Deque[AsyncWebCrawler] = deque()
        self.in_use: Set[AsyncWebCrawler] = set()
        self.last_used: Dict[AsyncWebCrawler, float] = {}
        self.cond = asyncio.Condition()


POOLS: Dict[str, _Pool] = {}
CRAWLER_SIG: Dict[AsyncWebCrawler, str] = {}


def _sig(cfg: BrowserConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def _get_pool(sig: str) -> _Pool:
    pool = POOLS.get(sig)
    if pool is None:
        pool = _Pool(max_size=POOL_MAX)
        POOLS[sig] = pool
    return pool


async def _create_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
    await crawler.start()
    return crawler


async def _close_crawler(crawler: AsyncWebCrawler, reason: str) -> None:
    logger.warning("Closing browser (%s)", reason)
    with suppress(Exception):
        await crawler.close()


async def checkout_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    sig = _sig(cfg)
    pool = _get_pool(sig)

    while True:
        async with pool.cond:
            if pool.available:
                crawler = pool.available.popleft()
                pool.in_use.add(crawler)
                return crawler

            if pool.total < pool.max_size:
                mem_pct = get_container_memory_percent()
                if mem_pct >= MEM_LIMIT:
                    raise MemoryError(f"Memory at {mem_pct:.1f}%, refusing new browser")
                pool.total += 1
                break

            await pool.cond.wait()

    try:
        crawler = await _create_crawler(cfg)
    except Exception:
        async with pool.cond:
            pool.total -= 1
            pool.cond.notify()
        raise

    async with pool.cond:
        pool.in_use.add(crawler)
        CRAWLER_SIG[crawler] = sig
    return crawler


async def return_crawler(crawler: AsyncWebCrawler) -> None:
    sig = CRAWLER_SIG.get(crawler)
    if not sig:
        await _close_crawler(crawler, "orphan return")
        return

    pool = POOLS.get(sig)
    if not pool:
        await _close_crawler(crawler, "pool missing")
        return

    if not getattr(crawler, "ready", False):
        await invalidate_crawler(crawler, reason="crawler not ready on return")
        return

    async with pool.cond:
        if crawler in pool.in_use:
            pool.in_use.remove(crawler)
        pool.available.append(crawler)
        pool.last_used[crawler] = time.time()
        pool.cond.notify()


async def invalidate_crawler(crawler: AsyncWebCrawler, *, reason: str = "unhealthy") -> None:
    sig = CRAWLER_SIG.pop(crawler, None)
    if not sig:
        await _close_crawler(crawler, reason)
        return

    pool = POOLS.get(sig)
    if pool is None:
        await _close_crawler(crawler, reason)
        return

    async with pool.cond:
        removed = False
        if crawler in pool.in_use:
            pool.in_use.remove(crawler)
            removed = True
        else:
            try:
                pool.available.remove(crawler)
                removed = True
            except ValueError:
                pass

        if removed:
            pool.total -= 1

        pool.last_used.pop(crawler, None)
        pool.cond.notify()

    await _close_crawler(crawler, reason)


async def init_permanent(cfg: BrowserConfig) -> None:
    """Pre-warm one crawler for the default config."""
    crawler = await checkout_crawler(cfg)
    await return_crawler(crawler)


async def get_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Backward-compatible alias (checkout). Prefer checkout_crawler/return_crawler."""
    return await checkout_crawler(cfg)


async def close_all():
    for sig, pool in list(POOLS.items()):
        to_close = []
        async with pool.cond:
            to_close.extend(list(pool.available))
            to_close.extend(list(pool.in_use))
            pool.available.clear()
            pool.in_use.clear()
            pool.last_used.clear()
            pool.total = 0
            pool.cond.notify_all()
        for crawler in to_close:
            await _close_crawler(crawler, "close_all")
    POOLS.clear()
    CRAWLER_SIG.clear()


async def janitor():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sig, pool in list(POOLS.items()):
            to_close = []
            async with pool.cond:
                for crawler in list(pool.available):
                    last = pool.last_used.get(crawler, 0)
                    if now - last > IDLE_TTL:
                        try:
                            pool.available.remove(crawler)
                        except ValueError:
                            continue
                        pool.last_used.pop(crawler, None)
                        pool.total -= 1
                        to_close.append(crawler)
                if to_close:
                    pool.cond.notify_all()
            for crawler in to_close:
                await _close_crawler(crawler, "idle timeout")
