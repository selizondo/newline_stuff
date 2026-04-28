"""
Langfuse results dashboard for the synthetic_data_diy pipeline.

Reads from the Langfuse HTTP API — works with self-hosted instances.
Uses Basic Auth (public_key:secret_key) so no SDK version dependency.

Usage:
  python langfuse_dash.py                          # All sessions overview
  python langfuse_dash.py --session baseline-mock  # Drill into one session
  python langfuse_dash.py --limit 500              # Fetch more items (default: 200)
  python langfuse_dash.py --since 24h              # Filter last 24h (or 7d, 30d)
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(auth: tuple, host: str, path: str, params: dict | None = None, no_auth: bool = False) -> dict:
    kwargs: dict = {"params": params or {}, "timeout": 10}
    if not no_auth:
        kwargs["auth"] = auth
    resp = requests.get(f"{host}{path}", **kwargs)
    resp.raise_for_status()
    return resp.json()


def _fetch_all(auth: tuple, host: str, path: str, params: dict | None = None, max_items: int = 200) -> list[dict]:
    results: list[dict] = []
    page = 1
    p = dict(params or {})
    while True:
        p["page"] = page
        p["limit"] = min(100, max_items - len(results))
        data = _get(auth, host, path, p)
        batch = data.get("data", [])
        results.extend(batch)
        total = data.get("meta", {}).get("totalItems", 0)
        if not batch or len(results) >= total or len(results) >= max_items:
            break
        page += 1
    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{text}\n{line}")


def _section(text: str) -> None:
    print(f"\n{'─' * 50}\n{text}\n{'─' * 50}")


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "     —"
    if ms < 1000:
        return f"{ms:>5.0f}ms"
    return f"{ms/1000:>5.1f}s "


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%m-%d %H:%M")
    except Exception:
        return ts[:16]


def _parse_since(since_str: str) -> datetime | None:
    if not since_str:
        return None
    unit = since_str[-1]
    if unit not in ("h", "d") or not since_str[:-1].isdigit():
        print(f"WARNING: unrecognized --since format '{since_str}' (use e.g. 24h, 7d) — ignoring")
        return None
    n = int(since_str[:-1])
    delta = timedelta(hours=n) if unit == "h" else timedelta(days=n)
    return datetime.now(timezone.utc) - delta


def _obs_latency_ms(obs: dict) -> float | None:
    lat = obs.get("latency")
    return lat * 1000 if lat is not None else None


# ── Overview ──────────────────────────────────────────────────────────────────

def _overview(auth: tuple, host: str, max_items: int, since: datetime | None) -> None:
    try:
        health = _get(auth, host, "/api/public/health", no_auth=True)
    except Exception:
        health = {"status": "unknown (health endpoint unreachable)"}

    trace_params: dict = {}
    if since:
        trace_params["fromTimestamp"] = since.isoformat()
    traces = _fetch_all(auth, host, "/api/public/traces", trace_params, max_items)

    obs_params: dict = {"type": "GENERATION"}
    if since:
        obs_params["fromStartTime"] = since.isoformat()
    observations = _fetch_all(auth, host, "/api/public/observations", obs_params, max_items)

    # Index observations by trace_id
    obs_by_trace: dict[str, list] = defaultdict(list)
    for o in observations:
        obs_by_trace[o["traceId"]].append(o)

    # Group traces by sessionId
    by_session: dict[str, list] = defaultdict(list)
    for t in traces:
        sid = t.get("sessionId") or "(no session)"
        by_session[sid].append(t)

    _banner("LANGFUSE DASHBOARD — synthetic_data_diy")
    print(f"Host        : {host}")
    print(f"Status      : {health.get('status', '?')}")
    since_str = f"  (since {since.strftime('%Y-%m-%d %H:%M UTC')})" if since else ""
    print(f"Traces      : {len(traces)}{since_str}")
    print(f"Generations : {len(observations)}")
    print(f"Sessions    : {len(by_session)}")

    if not traces:
        print("\nNo traces found. Run the pipeline (phases 1–7) to populate.")
        print("Verify langfuse=True appears in startup output when running main.py.")
        return

    _section("Sessions overview")
    col_w = max((len(s) for s in by_session), default=20) + 2
    col_w = max(col_w, 22)
    hdr = f"  {'Session':<{col_w}}  {'Traces':>6}  {'Gens':>5}  {'Errors':>6}  {'AvgLat':>8}  {'Tokens':>7}  Last"
    print(hdr)
    print("─" * (len(hdr) + 4))

    for sid, trace_list in sorted(by_session.items()):
        trace_ids = {t["id"] for t in trace_list}
        gens = [o for tid in trace_ids for o in obs_by_trace.get(tid, [])]
        errors = sum(1 for o in gens if o.get("level") == "ERROR")
        lats = [_obs_latency_ms(o) for o in gens if _obs_latency_ms(o) is not None]
        avg_lat = sum(lats) / len(lats) if lats else None
        tokens = sum((o.get("usage") or {}).get("total", 0) or 0 for o in gens)
        last_ts = max((t.get("timestamp", "") for t in trace_list), default="")
        print(
            f"  {sid:<{col_w}}  {len(trace_list):>6}  {len(gens):>5}"
            f"  {errors if errors else '—':>6}  {_fmt_ms(avg_lat):>8}"
            f"  {tokens if tokens else '—':>7}  {_fmt_ts(last_ts)}"
        )

    print(f"\n  Run with --session <name> to drill into a specific session.")


# ── Session detail ────────────────────────────────────────────────────────────

def _session_detail(auth: tuple, host: str, session_id: str, max_items: int) -> None:
    traces = _fetch_all(auth, host, "/api/public/traces", {"sessionId": session_id}, max_items)
    if not traces:
        print(f"No traces found for session '{session_id}'.")
        print("Check the session name matches the batch_label used when running the pipeline.")
        return

    _banner(f"SESSION: {session_id}")

    # Collect all generation observations for these traces
    all_obs: list[dict] = []
    for t in traces:
        obs = _fetch_all(auth, host, "/api/public/observations",
                         {"traceId": t["id"], "type": "GENERATION"}, max_items=500)
        all_obs.extend(obs)

    errors_total = sum(1 for o in all_obs if o.get("level") == "ERROR")
    tokens_total = sum((o.get("usage") or {}).get("total", 0) or 0 for o in all_obs)
    lats = [_obs_latency_ms(o) for o in all_obs if _obs_latency_ms(o) is not None]

    print(f"Traces       : {len(traces)}")
    print(f"Generations  : {len(all_obs)}")
    print(f"Errors       : {errors_total}")
    print(f"Total tokens : {tokens_total}")
    print(f"Avg latency  : {_fmt_ms(sum(lats)/len(lats) if lats else None).strip()}")

    if not all_obs:
        print("\nNo generations recorded. Phase 1 mock runs don't call the LLM so produce no generations.")
        print("Run a real LLM phase (e.g. zero_shot) to see generation data.")
        return

    # ── Per-name breakdown ───────────────────────────────────────────────────
    _section("Generations by phase/name")
    by_name: dict[str, list] = defaultdict(list)
    for o in all_obs:
        by_name[o.get("name", "unknown")].append(o)

    col_w = max(len(n) for n in by_name) + 2
    hdr = f"  {'Name':<{col_w}}  {'N':>4}  {'Err':>4}  {'AvgLat':>8}  {'MinLat':>8}  {'MaxLat':>8}  {'Tokens':>7}"
    print(hdr)
    print("─" * len(hdr))

    for name, obs_list in sorted(by_name.items()):
        errs = sum(1 for o in obs_list if o.get("level") == "ERROR")
        ms_list = [_obs_latency_ms(o) for o in obs_list if _obs_latency_ms(o) is not None]
        avg = sum(ms_list) / len(ms_list) if ms_list else None
        tokens = sum((o.get("usage") or {}).get("total", 0) or 0 for o in obs_list)
        print(
            f"  {name:<{col_w}}  {len(obs_list):>4}  {errs if errs else '—':>4}"
            f"  {_fmt_ms(avg):>8}  {_fmt_ms(min(ms_list) if ms_list else None):>8}"
            f"  {_fmt_ms(max(ms_list) if ms_list else None):>8}  {tokens if tokens else '—':>7}"
        )

    # ── Model usage ──────────────────────────────────────────────────────────
    models: dict[str, int] = defaultdict(int)
    for o in all_obs:
        if o.get("model"):
            models[o["model"]] += 1
    if models:
        _section("Model usage")
        for model, count in sorted(models.items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} call(s)")

    # ── Category distribution (from trace metadata) ──────────────────────────
    cats: dict[str, int] = defaultdict(int)
    for t in traces:
        cat = (t.get("metadata") or {}).get("category")
        if cat:
            cats[cat] += 1
    if cats:
        _section("Category distribution")
        total_cats = sum(cats.values())
        for cat, count in sorted(cats.items()):
            bar = "█" * int(count / total_cats * 20)
            print(f"  {cat:<28} {count:>3} ({count/total_cats*100:.0f}%)  {bar}")

    # ── Recent errors ────────────────────────────────────────────────────────
    error_obs = [o for o in all_obs if o.get("level") == "ERROR"]
    if error_obs:
        _section(f"Errors ({len(error_obs)})")
        for o in error_obs[:10]:
            msg = (o.get("statusMessage") or "")[:80]
            print(f"  [{o.get('name', '?')}] {msg or '(no message)'}")
        if len(error_obs) > 10:
            print(f"  ... and {len(error_obs) - 10} more")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Langfuse dashboard for synthetic_data_diy pipeline")
    parser.add_argument("--session", type=str, default=None,
                        help="Drill into a specific session (batch_label)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max items to fetch per resource type (default: 200)")
    parser.add_argument("--since", type=str, default=None,
                        help="Filter by recency: 24h, 7d, 30d")
    args = parser.parse_args()

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")

    if not pk or not sk:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        sys.exit(1)

    auth = (pk, sk)
    since = _parse_since(args.since) if args.since else None

    try:
        if args.session:
            _session_detail(auth, host, args.session, args.limit)
        else:
            _overview(auth, host, args.limit, since)
    except requests.HTTPError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"Cannot connect to Langfuse at {host}")
        sys.exit(1)


if __name__ == "__main__":
    main()
