"""
measure_latency.py -- Measures the deployed API's latency from this machine.

    python scripts/measure_latency.py https://museum-qa-api-xxxx.run.app
    python scripts/measure_latency.py URL --cold-wait 1200   # wait 20 min idle first

Reports what it measured and under what condition -- never an estimate:
  - first request : the first request this run made. It is a COLD START only
                    if the service had scaled to zero; Cloud Run scales an
                    idle service to zero after roughly 15 minutes, so use
                    --cold-wait (or run it long after the last request).
  - warm          : N further requests, after the first, as p50 / p90 / max.

Each request is a POST /evaluate on one recorded sample, the same call the
dashboard makes. Sends nothing sensitive; makes no model call (the API cannot).
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


def timed(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
        status = r.status
    return (time.perf_counter() - t0) * 1000, status, payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("--sample", default="q01")
    p.add_argument("-n", type=int, default=20, help="warm requests after the first")
    p.add_argument("--cold-wait", type=int, default=0,
                   help="seconds to stay idle before the first request")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    url = base + "/evaluate"
    body = {"sample_id": args.sample}

    if args.cold_wait:
        print(f"Idle for {args.cold_wait}s so the service can scale to zero...")
        time.sleep(args.cold_wait)

    first_ms, status, payload = timed(url, body)
    print(f"first request : {first_ms:7.0f} ms  (HTTP {status}, server compute "
          f"{payload.get('evaluated_in_ms')} ms)")

    warm = []
    for _ in range(args.n):
        ms, status, _ = timed(url, body)
        if status != 200:
            sys.exit(f"warm request failed: HTTP {status}")
        warm.append(ms)
        time.sleep(0.3)          # stay well under the 30/min rate limit window
    warm.sort()
    q = statistics.quantiles(warm, n=10)
    print(f"warm (n={len(warm)}): p50 {statistics.median(warm):.0f} ms, "
          f"p90 {q[8]:.0f} ms, max {warm[-1]:.0f} ms")
    print("Note: 'first request' is a cold start only if the service had scaled to zero.")


if __name__ == "__main__":
    main()
