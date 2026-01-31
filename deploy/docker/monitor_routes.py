# monitor_routes.py - Monitor API endpoints
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from monitor import get_monitor
import logging
import asyncio
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("/health")
async def get_health():
    """Get current system health snapshot."""
    try:
        monitor = get_monitor()
        return await monitor.get_health_summary()
    except Exception as e:
        logger.error(f"Error getting health: {e}")
        raise HTTPException(500, str(e))


@router.get("/requests")
async def get_requests(status: str = "all", limit: int = 50):
    """Get active and completed requests.

    Args:
        status: Filter by 'active', 'completed', 'success', 'error', or 'all'
        limit: Max number of completed requests to return (default 50)
    """
    # Input validation
    if status not in ["all", "active", "completed", "success", "error"]:
        raise HTTPException(400, f"Invalid status: {status}. Must be one of: all, active, completed, success, error")
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()

        if status == "active":
            return {"active": monitor.get_active_requests(), "completed": []}
        elif status == "completed":
            return {"active": [], "completed": monitor.get_completed_requests(limit)}
        elif status in ["success", "error"]:
            return {"active": [], "completed": monitor.get_completed_requests(limit, status)}
        else:  # "all"
            return {
                "active": monitor.get_active_requests(),
                "completed": monitor.get_completed_requests(limit)
            }
    except Exception as e:
        logger.error(f"Error getting requests: {e}")
        raise HTTPException(500, str(e))


@router.get("/browsers")
async def get_browsers():
    """Get detailed browser pool information."""
    try:
        monitor = get_monitor()
        browsers = await monitor.get_browser_list()

        # Calculate summary stats
        total_browsers = len(browsers)
        total_memory = sum(b["memory_mb"] for b in browsers)

        # Calculate reuse rate from recent requests
        recent = monitor.get_completed_requests(100)
        pool_hits = sum(1 for r in recent if r.get("pool_hit", False))
        reuse_rate = (pool_hits / len(recent) * 100) if recent else 0

        return {
            "browsers": browsers,
            "summary": {
                "total_count": total_browsers,
                "total_memory_mb": total_memory,
                "reuse_rate_percent": round(reuse_rate, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error getting browsers: {e}")
        raise HTTPException(500, str(e))


@router.get("/endpoints/stats")
async def get_endpoint_stats():
    """Get aggregated endpoint statistics."""
    try:
        monitor = get_monitor()
        return monitor.get_endpoint_stats_summary()
    except Exception as e:
        logger.error(f"Error getting endpoint stats: {e}")
        raise HTTPException(500, str(e))


@router.get("/timeline")
async def get_timeline(metric: str = "memory", window: str = "5m"):
    """Get timeline data for charts.

    Args:
        metric: 'memory', 'requests', or 'browsers'
        window: Time window (only '5m' supported for now)
    """
    # Input validation
    if metric not in ["memory", "requests", "browsers"]:
        raise HTTPException(400, f"Invalid metric: {metric}. Must be one of: memory, requests, browsers")
    if window != "5m":
        raise HTTPException(400, f"Invalid window: {window}. Only '5m' is currently supported")

    try:
        monitor = get_monitor()
        return monitor.get_timeline_data(metric, window)
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(500, str(e))


@router.get("/logs/janitor")
async def get_janitor_log(limit: int = 100):
    """Get recent janitor cleanup events."""
    # Input validation
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()
        return {"events": monitor.get_janitor_log(limit)}
    except Exception as e:
        logger.error(f"Error getting janitor log: {e}")
        raise HTTPException(500, str(e))


@router.get("/logs/errors")
async def get_errors_log(limit: int = 100):
    """Get recent errors."""
    # Input validation
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()
        return {"errors": monitor.get_errors_log(limit)}
    except Exception as e:
        logger.error(f"Error getting errors log: {e}")
        raise HTTPException(500, str(e))


# ========== Control Actions ==========

class KillBrowserRequest(BaseModel):
    sig: str


@router.post("/actions/cleanup")
async def force_cleanup():
    """Force immediate janitor cleanup (kills idle cold pool browsers)."""
    try:
        from crawler_pool import close_available
        killed_count = await close_available(reason="force cleanup")

        monitor = get_monitor()
        await monitor.track_janitor_event("force_cleanup", "manual", {"killed": killed_count})

        return {"success": True, "killed_browsers": killed_count}
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")
        raise HTTPException(500, str(e))


@router.post("/actions/kill_browser")
async def kill_browser(req: KillBrowserRequest):
    """Kill a specific browser by signature (hot or cold only).

    Args:
        sig: Browser config signature (first 8 chars)
    """
    try:
        from crawler_pool import kill_by_sig_prefix

        result = await kill_by_sig_prefix(req.sig)
        if result["killed"] == 0 and result["in_use"] > 0:
            raise HTTPException(409, "All matching browsers are currently in use")
        if result["killed"] == 0:
            raise HTTPException(404, f"Browser with sig={req.sig} not found")

        monitor = get_monitor()
        await monitor.track_janitor_event("kill_browser", req.sig, {"manual": True, **result})

        return {"success": True, "killed_sig": req.sig, **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error killing browser: {e}")
        raise HTTPException(500, str(e))


@router.post("/actions/restart_browser")
async def restart_browser(req: KillBrowserRequest):
    """Restart a browser (kill + recreate). Works for permanent too.

    Args:
        sig: Browser config signature (first 8 chars), or "permanent"
    """
    try:
        from crawler_pool import kill_by_sig_prefix

        result = await kill_by_sig_prefix(req.sig)
        if result["killed"] == 0 and result["in_use"] > 0:
            raise HTTPException(409, "Matching browsers are in use; retry later")
        if result["killed"] == 0:
            raise HTTPException(404, f"Browser with sig={req.sig} not found")

        monitor = get_monitor()
        await monitor.track_janitor_event("restart_browser", req.sig, {"manual": True, **result})

        return {"success": True, "restarted_sig": req.sig, "note": "Browser will be recreated on next request"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting browser: {e}")
        raise HTTPException(500, str(e))


@router.post("/stats/reset")
async def reset_stats():
    """Reset today's endpoint counters."""
    try:
        monitor = get_monitor()
        monitor.endpoint_stats.clear()
        await monitor._persist_endpoint_stats()

        return {"success": True, "message": "Endpoint stats reset"}
    except Exception as e:
        logger.error(f"Error resetting stats: {e}")
        raise HTTPException(500, str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates.

    Sends updates every 2 seconds with:
    - Health stats
    - Active/completed requests
    - Browser pool status
    - Timeline data
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            try:
                # Gather all monitoring data
                monitor = get_monitor()

                data = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "health": await monitor.get_health_summary(),
                    "requests": {
                        "active": monitor.get_active_requests(),
                        "completed": monitor.get_completed_requests(limit=10)
                    },
                    "browsers": await monitor.get_browser_list(),
                    "timeline": {
                        "memory": monitor.get_timeline_data("memory", "5m"),
                        "requests": monitor.get_timeline_data("requests", "5m"),
                        "browsers": monitor.get_timeline_data("browsers", "5m")
                    },
                    "janitor": monitor.get_janitor_log(limit=10),
                    "errors": monitor.get_errors_log(limit=10)
                }

                # Send update to client
                await websocket.send_json(data)

                # Wait 2 seconds before next update
                await asyncio.sleep(2)

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await asyncio.sleep(2)  # Continue trying

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")
