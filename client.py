"""Make one picture: wake a worker, POST /generate, save the JPEG and print the stage timings."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Worker URL printed by modal deploy")
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="image.jpg")
    args = parser.parse_args()
    headers = {
        "Modal-Key": os.environ["MODAL_KEY"],
        "Modal-Secret": os.environ["MODAL_SECRET"],
        "Content-Type": "application/json",
    }
    base = args.url.rstrip("/")
    started = time.perf_counter()
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/wake", b"", headers), timeout=30):
                break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (502, 503, 504):
                sys.exit(f"wake failed: {exc.code} {exc.read().decode()[:200]}")
            if time.perf_counter() - started > 900:
                sys.exit("no worker became ready within 15 minutes")
            time.sleep(5)
    print(f"ready after {time.perf_counter() - started:.1f} s", flush=True)
    body = json.dumps({"prompt": args.prompt, "seed": args.seed}).encode()
    sent = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/generate", body, headers), timeout=330
        ) as r:
            data, stages = r.read(), r.headers
    except urllib.error.HTTPError as exc:
        sys.exit(f"generate failed: {exc.code} {exc.read().decode()[:500]}")
    with open(args.out, "wb") as f:
        f.write(data)
    print(
        f"{args.out}: {len(data)} bytes in {(time.perf_counter() - sent) * 1000:.0f} ms "
        f"(upsampler {float(stages['X-Pup-Ms']):.0f} ms, of which inside the upsampler {float(stages['X-Upsampler-Ms']):.0f} ms; "
        f"image queue {float(stages['X-Image-Queue-Ms']):.0f} ms, image {float(stages['X-Image-Ms']):.0f} ms)"
    )


if __name__ == "__main__":
    main()
