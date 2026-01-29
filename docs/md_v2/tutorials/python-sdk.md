# Crawl4AI Python SDK 市场分析教学

> 目标：把任何公开网站转成可用于竞争对手研究、价格跟踪与产品摘要的结构化数据。文档从零开始，逐步引导你配置浏览器、编写异步脚本、执行多页/深度爬取，并利用 LLM 与表格策略提炼市场情报。

## 0. 阅读前提与速览
- Python >= 3.10，能够读懂函数、类以及基本 `async`/`await` 语法。
- 具备虚拟环境与 `pip` 使用经验；不了解也没关系，本节会解释。
- 了解基础 Markdown、JSON 即可，复杂数据结构会配合示例说明。
- 推荐在独立 virtualenv/conda 环境中操作，避免与系统包冲突。

**官方安装命令**：

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -U crawl4ai
crawl4ai-setup          # 安装 Playwright 浏览器依赖
crawl4ai-doctor         # 快速健康检查
python -m playwright install --with-deps chromium
```

`crawl4ai-setup` 会生成默认缓存与数据库目录；`crawl4ai-doctor` 可提前发现显卡、权限或浏览器缺失问题。[^install-readme]

---

## 1. 核心概念速记卡

| 组件 | 职责 | 要点 |
| --- | --- | --- |
| `AsyncWebCrawler` | 统一的异步浏览器控制器 | 可作为 `async with` 上下文，提供 `arun` (单源)、`arun_many` (多源) 等方法。[^asyncwebcrawler] |
| `BrowserConfig` | 定义浏览器形态、代理、指纹 | 支持 headless/可视化、CDP/Docker、持久化上下文、UA 随机、stealth、文本模式等。[^browser-config] |
| `CrawlerRunConfig` | 描述每次爬取应如何处理页面 | 内容过滤、JS 交互、截图/PDF、深度爬、虚拟滚动、链接过滤、缓存策略等集中在此。[^crawler-run-config][^browser-config-md] |
| `CacheMode` & `CacheContext` | 控制缓存读写 | `ENABLED`、`BYPASS`、`READ_ONLY`、`WRITE_ONLY` 等枚举比旧式布尔标志更清晰。[^cache-mode] |
| `CrawlResult` | 返回结构体 | 含 `markdown`、`fit_markdown`、`extracted_content`、`tables`、截图/PDF、网络日志等，可直接送入你现有的数据管道。[^crawl-result] |

**Python 高阶概念回顾**

- `async def` 定义协程函数；`await` 暂停并让事件循环调度其它任务。
- `async with AsyncWebCrawler()` 是“异步上下文管理器”：进入时启动 Playwright，退出时关闭浏览器，避免资源泄露。
- `dataclass` / `pydantic` 用于描述 schema，让提取结果拥有类型提示与验证。
- `asyncio.gather` 可并发等待多个 `awaitable`，常用于批量 URL 爬取。

---

## 2. Hello Crawl4AI：第一个脚本

```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_once() -> None:
    browser_cfg = BrowserConfig(
        headless=True,
        enable_stealth=False,
        text_mode=False,
    )

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,          # 强制重新下载，确保调试一致
        wait_for="css:article",               # 等待主内容渲染
        screenshot=True,
        verbose=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=run_cfg,
        )

        # StringCompatibleMarkdown 既能当字符串切片，也能访问属性
        print(result.markdown.raw_markdown[:400])
        print("截图大小(B):", len(result.screenshot or ""))

if __name__ == "__main__":
    asyncio.run(crawl_once())
```

运行要点：

1. `asyncio.run` 会创建事件循环 → 逐步执行 `crawl_once`。
2. `result.markdown` 是一个“像字符串一样”的对象，继续 `.raw_markdown`、`.fit_markdown`、`.references_markdown` 可直接喂给 RAG。[^crawl-result]
3. 如果站点只在特定区域渲染内容，可配置 `CrawlerRunConfig.css_selector` 或 `target_elements` 把 DOM 缩小到正文区域。[^crawler-run-config]

---

## 3. 浏览器身份与反爬策略 (BrowserConfig)

`BrowserConfig` 决定了 Playwright 浏览器如何被启动，典型场景：[^browser-config]

```python
from crawl4ai import BrowserConfig, ProxyConfig
from crawl4ai.proxy_strategy import RoundRobinProxyStrategy

proxy_strategy = RoundRobinProxyStrategy([
    ProxyConfig.from_string("12.34.56.78:8000:user:pass"),
    ProxyConfig.from_string("98.76.54.32:9000"),
])

browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=False,                # 调试期可见
    enable_stealth=True,           # playwright-stealth，隐藏 webdriver 指纹
    use_persistent_context=True,
    user_data_dir="./profiles/retail-us",
    user_agent_mode="random",
    viewport_width=1440,
    viewport_height=900,
    text_mode=True,                # 纯文本可提速
)
```

在 `CrawlerRunConfig` 中配置 `proxy_rotation_strategy=proxy_strategy` 即可在请求之间轮换代理。[^proxy-doc]

实战建议：

1. **持久化上下文**：登录后把 cookies 存在 `user_data_dir`，下次直接复用，避免重复登录。
2. **代理 & 指纹**：结合 `enable_stealth` + 代理池降低被封概率；必要时切换到 `browser_mode="cdp"` 重用线上浏览器。
3. **地理/语言**：`CrawlerRunConfig(locale="en-US", timezone_id="America/New_York", geolocation=GeolocationConfig(...))` 可模拟不同地区的价格/库存。[^crawler-run-config]

---

## 4. CrawlerRunConfig：让页面按需输出

`CrawlerRunConfig` 把“这一页需要什么内容、如何等待、是否截图、是否执行 JS”等全部封装起来。[^browser-config-md][^arun-doc]

### 4.1 内容裁剪与清洗
```python
CrawlerRunConfig(
    css_selector="main article",
    target_elements=[".pricing-card", ".hero"],
    excluded_tags=["form", "nav"],
    remove_forms=True,
    word_count_threshold=30,
    markdown_generator=DefaultMarkdownGenerator(),
)
```
- `css_selector` 直接缩小 DOM，再传给 Markdown/提取流程。
- `target_elements` 仅影响提取；`css_selector` 是“真裁剪”。
- `word_count_threshold` 防止导航栏、脚注混入。

### 4.2 加载控制
```python
CrawlerRunConfig(
    wait_for="js:() => document.querySelectorAll('.product-card').length >= 12",
    wait_for_timeout=15_000,
    js_code=[
        "document.querySelector('.cookie-accept')?.click()",
        "document.querySelector('.load-more')?.click()",
    ],
    js_only=False,
    virtual_scroll_config=VirtualScrollConfig(
        container_selector="[data-virtual-list]",
        scroll_count=8,
    ),
    simulate_user=True,
    magic=True,
)
```
- `wait_for` 支持 `css:` 与 `js:` 前缀，适合动态站。[^arun-doc]
- `js_code` 可执行任意脚本，复杂交互可使用 `c4a-script`。
- `virtual_scroll_config` 自动滚动虚拟列表，例如股票行情或招聘列表。[^virtual-scroll]

### 4.3 媒体输出
设置 `screenshot=True`, `pdf=True`, `capture_mhtml=True` 可保留证据。`image_score_threshold` 与 `exclude_external_images` 可过滤广告图。

### 4.4 缓存与会话
`cache_mode=CacheMode.ENABLED` 默认读写缓存；若需要保证“今天的价格”，使用 `CacheMode.BYPASS`。批量任务中，可将 `session_id="nike-us"`，在复杂交互后复用同一个标签页。[^cache-mode][^crawler-run-config]

---

## 5. 批量 / 深度爬取策略

### 5.1 `arun_many` 并发
```python
urls = [
    "https://shop.example.com/new",
    "https://shop.example.com/sale",
    # ...
]

async with AsyncWebCrawler(config=browser_cfg) as crawler:
    results = await crawler.arun_many(
        sources=urls,
        config=run_cfg,
        semaphore_count=5,    # 控制并发
        stream=True,          # 边爬边 yield
    )
    async for page in results:
        handle(page)
```
`stream=True` 时会返回异步迭代器，可在首个页面完成时就开始处理。[^arun-doc]

### 5.2 深度爬策略
- `BFSDeepCrawlStrategy` / `DFSDeepCrawlStrategy`：设置 `max_depth`, `include_external`, `max_pages` 控制链接扩散。[^deep-crawl-doc]
- `BestFirstCrawlingStrategy` + `KeywordRelevanceScorer`：优先访问“可能含有价格/新品”的 URL。
- `FilterChain`：用 URL 通配符、域名白名单等过滤不相关页面。

### 5.3 URL 发现
`AsyncUrlSeeder` 可以从 sitemap、Common Crawl 拉取 URL，支持 BM25 评分、HEAD 检查与 JSONL 缓存，适合大站点初筛。[^url-seeder]

### 5.4 自适应抓取
`AdaptiveCrawler` 根据 Coverage / Consistency / Saturation 三个指标自动决定是否继续深爬；非常适合“找到足够多的同类产品就停”。[^adaptive-doc]

---

## 6. 将网页转成市场情报

### 6.1 无 LLM 的 schema 抽取
结构稳定的站点优先使用 `JsonCssExtractionStrategy` 或 `JsonXPathExtractionStrategy`，既快又省钱：[^no-llm]

```python
schema = {
    "name": "Product Cards",
    "baseSelector": "div.product-card",
    "fields": [
        {"name": "title", "selector": "h3", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"},
    ],
}

config = CrawlerRunConfig(
    extraction_strategy=JsonCssExtractionStrategy(schema, verbose=True),
    cache_mode=CacheMode.BYPASS,
)
```

### 6.2 LLM 提取（适合文案、卖点）
当页面结构不稳定或需要摘要时，使用 `LLMExtractionStrategy` + `LLMConfig`：[^llm-strategy-doc][^llm-config-code]

```python
from pydantic import BaseModel, Field
from crawl4ai import LLMConfig, LLMExtractionStrategy

class ProductCard(BaseModel):
    brand: str = Field(..., description="品牌名")
    product: str = Field(..., description="商品名称")
    price: str
    highlights: list[str]
    url: str

llm_strategy = LLMExtractionStrategy(
    llm_config=LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token="env:OPENAI_API_KEY",
        temperature=0.1,
    ),
    schema=ProductCard.model_json_schema(),
    extraction_type="schema",
    instruction=(
        "读取页面 Markdown，输出包含 brand, product, price, highlights, url "
        "字段的 JSON 数组，金额保留原货币。"
    ),
    chunk_token_threshold=1200,
    overlap_rate=0.1,
    input_format="markdown",
    extra_args={"max_tokens": 800},
)
```

把该策略塞进 `CrawlerRunConfig(extraction_strategy=llm_strategy)` 即可。记得在高成本模型时配合 `CacheMode.ENABLED` 或调用 `llm_strategy.show_usage()` 查看 token。

### 6.3 表格提取
- 默认 `DefaultTableExtraction` 会自动发现 `<table>`，把标题、caption、score 等放进 `result.tables`。[^table-doc]
- 如遇复杂跨行/跨列，才使用 `LLMTableExtraction`，可指定 `chunk_token_threshold` 与 `max_parallel_chunks` 控制成本。
- `table_score_threshold` 可以过滤“装饰性表格”，只留下得分 >=7 的真正数据表。

### 6.4 虚拟滚动与 JS 交互
对“无限列表”或 React 虚拟化页面，组合 `VirtualScrollConfig` 与 `js_code` 连续触发“加载更多”。滚动结束后再进行提取，可显著提升命中率。[^virtual-scroll][^arun-doc]

### 6.5 链接预览与打分
`link_preview_config` 可异步抓取所有 `<a>` 的 `head`，并用 BM25/关键词评分，将“值得进一步爬取的 URL”写进结果。适合市场分析中的“再下一层”策略。[^crawler-run-config]

---

## 7. 数据整理与分析

拿到 `CrawlResult` 后的典型处理流程：

```python
import json
import pandas as pd

records = []
for page in pages:
    data = json.loads(page.extracted_content or "[]")
    for item in data:
        records.append({
            "source": page.url,
            **item,
            "timestamp": page.metadata.get("timestamp"),
        })

df = pd.DataFrame(records)
df["price_value"] = (
    df["price"]
    .str.extract(r"([0-9.,]+)")
    .replace({",": ""}, regex=True)
    .astype(float)
)
print(df.groupby("brand")["price_value"].describe())
```

- `result.media`, `result.links`, `result.tables` 也都是列表字典，方便转成 DataFrame。
- `screenshot` / `pdf` 可存盘作为 audit trail。
- 所有字段均可追溯回 `CrawlResult`，保证数据血缘。[^crawl-result]

---

## 8. 端到端示例：多站点价格/卖点抓取

```python
import asyncio, json, base64
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel, Field

from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
    LLMConfig, LLMExtractionStrategy, DefaultTableExtraction,
)

PRODUCT_PAGES = [
    {"brand": "Nike", "url": "https://www.nike.com/w/new-mens-shoes"},
    {"brand": "Adidas", "url": "https://www.adidas.com/us/men-new_arrivals"},
    {"brand": "New Balance", "url": "https://www.newbalance.com/men/shoes/"},
]
brand_by_url = {entry["url"]: entry["brand"] for entry in PRODUCT_PAGES}

class MarketCard(BaseModel):
    brand: str = Field(..., description="品牌名称")
    product: str = Field(..., description="商品名称")
    price: str
    highlights: list[str] = Field(default_factory=list)
    url: str

llm_strategy = LLMExtractionStrategy(
    llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token="env:OPENAI_API_KEY"),
    schema=MarketCard.model_json_schema(),
    extraction_type="schema",
    instruction=(
        "从页面 Markdown 中找出所有在售鞋款。"
        "brand 使用页面真实品牌，price 保留货币符号，"
        "highlights 用 1-3 条 bullet 概括卖点。"
    ),
    chunk_token_threshold=1400,
    apply_chunking=True,
    input_format="markdown",
    extra_args={"temperature": 0.0, "max_tokens": 1200},
)

table_strategy = DefaultTableExtraction(
    table_score_threshold=8,
    min_rows=3,
    verbose=False,
)

browser_cfg = BrowserConfig(
    headless=True,
    enable_stealth=True,
    text_mode=False,
    viewport_width=1280,
    viewport_height=720,
)

base_run_cfg = CrawlerRunConfig(
    extraction_strategy=llm_strategy,
    table_extraction=table_strategy,
    cache_mode=CacheMode.BYPASS,
    wait_for="css:.product-card, .product-tile",
    js_code=[
        "document.querySelector('button[aria-label=\"Close\"]')?.click()",
        "document.querySelector('.load-more, .gl-load-more')?.click()",
    ],
    screenshot=True,
    verbose=False,
)

async def crawl_targets(targets: List[Dict[str, str]]):
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        tasks = []
        for t in targets:
            cfg = base_run_cfg.clone()
            tasks.append(crawler.arun(url=t["url"], config=cfg))
        return await asyncio.gather(*tasks)

def main():
    raw_results = asyncio.run(crawl_targets(PRODUCT_PAGES))
    structured_rows, tables = [], []
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    page_shots = {}

    for idx, res in enumerate(raw_results):
        data = json.loads(res.extracted_content or "[]")
        for item in data:
            structured_rows.append({
                "source": res.url,
                "brand": item.get("brand") or brand_by_url.get(res.url),
                "product": item.get("product"),
                "price": item.get("price"),
                "highlights": item.get("highlights", []),
                "url": item.get("url") or res.url,
                "screenshot_path": None,
            })
        tables.extend(res.tables)
        if res.screenshot:
            shot_path = artifact_dir / f"page_{idx}.png"
            shot_path.write_bytes(base64.b64decode(res.screenshot))
            page_shots[res.url] = str(shot_path)

    for row in structured_rows:
        row["screenshot_path"] = page_shots.get(row["source"])

    df = pd.DataFrame(structured_rows)
    df = df.explode("highlights", ignore_index=True)
    df.to_csv("market_cards.csv", index=False)

    tables_df = pd.DataFrame(tables)
    tables_df.to_json("captured_tables.json", orient="records", indent=2)

    print(df.head())
    print("Captured tables:", len(tables))

if __name__ == "__main__":
    main()
```

**代码拆解**

1. `LLMExtractionStrategy` + `Pydantic` 让输出字段固定，避免 AI 随意扩展。[^llm-strategy-doc]
2. `DefaultTableExtraction` 额外抓出“尺码对照/价格对比”表，可与 `MarketCard` 一起分析。[^table-doc]
3. `base_run_cfg.clone()` 是最常见的复用模式 —— 保持核心配置不变，对每个 URL 单独微调。[^crawler-run-config]
4. `asyncio.gather` 并行抓取多个品牌页面，速度能比同步脚本快约 3 倍。
5. `page_shots` 记录 URL → 截图路径，方便把分析结果回溯到原始画面。

---

## 9. 调试与最佳实践

1. **尊重 robots 与法律**：设置 `check_robots_txt=True`，确保遵守站点政策。[^arun-doc]
2. **日志**：`AsyncLogger` 默认写在 `~/.crawl4ai/crawler.log`。调试复杂 JS 时把 `verbose=True` 与 `log_console=True` 打开。
3. **Hooks 扩展**：`AsyncPlaywrightCrawlerStrategy` 暴露 `on_browser_created`、`before_goto`、`before_retrieve_html` 等钩子，可注入自定义逻辑（如打点、添加 headers）。[^hooks]
4. **缓存策略**：批量监控建议 `CacheMode.ENABLED`，临时验证使用 `CacheMode.BYPASS`，而写后不读可使用 `WRITE_ONLY` 加速。[^cache-mode]
5. **资源回收**：长时间运行时启用 `thread_safe=True` 或 `MemoryAdaptiveDispatcher`，并合理设置 `semaphore_count` 与 `mean_delay`，避免内存飙升。[^asyncwebcrawler]
6. **虚拟滚动/复杂交互**：优先使用 Playwright 脚本完成交互，再用 `CrawlerRunConfig` 提取；必要时可切换到 `UndetectedAdapter` 或 CDP 模式。
7. **成本控制**：LLM 策略配合 `cache_mode`, `chunk_token_threshold`、`input_format="fit_markdown"` 可显著降低 token 消耗。[^llm-strategy-doc]

---

## 10. 行动清单

1. ✅ 复制“Hello Crawl4AI”脚本，确认环境 OK。  
2. ✅ 根据目标站点设置 `BrowserConfig`（代理/地理/持久化）。  
3. ✅ 使用 `CrawlerRunConfig` 精准圈定内容区域，并记录需要的 JS、等待条件。  
4. ✅ 先尝试 `JsonCssExtractionStrategy`；若页面结构复杂再改用 LLM。  
5. ✅ 为竞品列表配置 `arun_many` 或 `DeepCrawlStrategy`，必要时配合 `AdaptiveCrawler` 提前收敛。  
6. ✅ 将 `CrawlResult` 转成 DataFrame，接入你已有的数据分析/可视化工具。  
7. ✅ 保存截图或 PDF，方便市场团队复核。  
8. ✅ 在正式上线前，用小范围 URL 列表跑通全流程，并检查缓存、速率、LLM 成本。

祝你构建出稳定、可复现、可审计的市场分析爬虫！

---

[^install-readme]: README.md:61-105.  
[^asyncwebcrawler]: crawl4ai/async_webcrawler.py:53-139.  
[^browser-config]: crawl4ai/async_configs.py:329-455.  
[^crawler-run-config]: crawl4ai/async_configs.py:833-1149.  
[^browser-config-md]: docs/md_v2/core/browser-crawler-config.md:1-210.  
[^cache-mode]: crawl4ai/cache_context.py:4-117.  
[^crawl-result]: crawl4ai/models.py:129-210.  
[^hooks]: crawl4ai/async_crawler_strategy.py:74-140.  
[^arun-doc]: docs/md_v2/api/arun.md:1-210.  
[^deep-crawl-doc]: docs/md_v2/core/deep-crawling.md:1-210.  
[^adaptive-doc]: docs/md_v2/core/adaptive-crawling.md:1-210.  
[^llm-strategy-doc]: docs/md_v2/extraction/llm-strategies.md:1-260.  
[^table-doc]: docs/md_v2/core/table_extraction.md:1-220.  
[^proxy-doc]: crawl4ai/proxy_strategy.py:1-158.  
[^virtual-scroll]: crawl4ai/async_configs.py:623-660.  
[^llm-config-code]: crawl4ai/async_configs.py:1670-1749.  
[^url-seeder]: crawl4ai/async_url_seeder.py:1-40.  
[^no-llm]: docs/md_v2/extraction/no-llm-strategies.md:1-220.
