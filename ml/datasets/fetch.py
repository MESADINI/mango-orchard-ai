"""
Robust downloads for public dataset files.

Mendeley Data serves files from S3 at only a few KB/s per connection from some networks, but throughput
scales with the number of connections. `parallel_download` fetches byte ranges concurrently, retries failed
chunks, resumes from previously completed chunks, and verifies the final SHA-256.
"""

import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

USER_AGENT = "mango-orchard-ai/1.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_url(url: str, timeout: float = 60) -> str:
    """Follows redirects once so every range request goes straight to the final host."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.geturl()


def content_length(url: str, timeout: float = 60) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Length"])


def _fetch_range(url: str, start: int, end: int, out: Path, retries: int, timeout: float) -> int:
    expected = end - start + 1
    if out.exists() and out.stat().st_size == expected:
        return expected
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) != expected:
                raise IOError(f"short read {len(data)}/{expected}")
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(out)
            return expected
        except Exception:  # network errors are retried with backoff
            time.sleep(min(60, 2 ** attempt))
    raise IOError(f"Range {start}-{end} failed after {retries} attempts")


def parallel_download(url: str, dest: Path, sha256: Optional[str] = None, workers: int = 64,
                      chunk_size: int = 1 << 20, retries: int = 8, timeout: float = 600) -> Path:
    dest = Path(dest)
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        print(f"Already downloaded: {dest}")
        return dest
    final_url = resolve_url(url)
    size = content_length(final_url)
    parts_dir = dest.with_name(dest.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "meta.json").write_text(json.dumps({"url": url, "size": size, "chunk": chunk_size}))
    ranges = [(i, s, min(s + chunk_size, size) - 1) for i, s in enumerate(range(0, size, chunk_size))]

    done, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_range, final_url, s, e, parts_dir / f"{i:06d}", retries, timeout)
                   for i, s, e in ranges]
        for k, fut in enumerate(as_completed(futures), 1):
            done += fut.result()
            if k % 25 == 0 or k == len(futures):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done / 1e6:.1f}/{size / 1e6:.1f} MB ({rate / 1e3:.0f} KB/s)", flush=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for i, _, _ in ranges:
            f.write((parts_dir / f"{i:06d}").read_bytes())
    if sha256 and sha256_file(tmp) != sha256:
        tmp.unlink()
        raise IOError(f"SHA-256 mismatch for {dest.name}")
    tmp.replace(dest)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    return dest
